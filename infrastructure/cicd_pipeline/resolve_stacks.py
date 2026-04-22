"""Resolve which CDK stacks to deploy for a given pipeline config.

Called at CodeBuild runtime (not CDK synth time) to dynamically discover
stacks based on the pipeline config YAML. Supports two phases:

  Phase 1 (consumer stacks): PipelineStack + A2IStack (if configured)
  Phase 2 (all stacks): DataStack + Phase 1 stacks

The shared CodeBuildStack is deployed separately by ``make deploy``
alongside CiCdPipelineStack and is NOT included here.

Usage:
  python3 resolve_stacks.py <config_file> <prefix> --phase 1|2
  python3 resolve_stacks.py <config_file> <prefix> --strategy
"""

import argparse

import boto3
import yaml
from botocore.exceptions import ClientError


def resolve_stacks(cfg_file: str, prefix: str, phase: int) -> list[str]:
    """Return the list of stack names to deploy for the given phase."""
    with open(f"config/pipeline/{cfg_file}") as f:
        cfg = yaml.safe_load(f)

    a2i = cfg.get("a2i") or {}
    ls = cfg.get("lambda_steps") or {}
    refs = {v.get("a2i_name") for v in ls.values() if v.get("a2i_name")}
    has_a2i = bool(refs & set(a2i.keys()))

    stacks = []
    if phase == 2:
        stacks.append(f"{prefix}-DataStack")
    if has_a2i:
        stacks.append(f"{prefix}-A2IStack")
    stacks.append(f"{prefix}-PipelineStack")
    return stacks


def _stack_exists(stack_name: str) -> bool:
    """Check if a CloudFormation stack exists and is in a usable state."""
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName=stack_name)
        status = resp["Stacks"][0]["StackStatus"]
        return status not in ("ROLLBACK_COMPLETE", "DELETE_COMPLETE", "DELETE_IN_PROGRESS")
    except ClientError:
        return False


def resolve_strategy(cfg_file: str, prefix: str) -> str:
    """Return 'combined' if any stack is new, else 'phased'."""
    all_stacks = resolve_stacks(cfg_file, prefix, phase=2)
    for stack in all_stacks:
        if not _stack_exists(stack):
            return "combined"
    return "phased"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("prefix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=[1, 2])
    group.add_argument("--strategy", action="store_true")
    args = parser.parse_args()

    if args.strategy:
        print(resolve_strategy(args.config_file, args.prefix))
    else:
        stacks = resolve_stacks(args.config_file, args.prefix, args.phase)
        print(" ".join(stacks))
