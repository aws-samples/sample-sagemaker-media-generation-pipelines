"""Generate the two-phase CDK deploy bash script for CodeBuild.

Phase 1 deploys consumer stacks (PipelineStack, A2IStack) to break
cross-stack CloudFormation imports. Phase 2 deploys all per-config stacks
so provider stacks can safely remove old exports.

If any resolved stack doesn't exist yet (e.g. A2IStack newly enabled),
falls back to a combined deploy so CDK handles dependency ordering.

The shared CodeBuildStack is deployed separately by ``make deploy``
alongside CiCdPipelineStack and is NOT included in per-config deploys.

Stack discovery is delegated to ``resolve_stacks.py`` which reads the
pipeline config YAML at CodeBuild runtime.
"""

SCRIPTS_DIR = "infrastructure/cicd_pipeline"


def generate_deploy_script(cfg_prefix: str, cfg_file: str, rollback_flag: str) -> str:
    """Return the full bash script for the Deploy CodeBuild action."""
    resolve = f"python3 {SCRIPTS_DIR}/resolve_stacks.py {cfg_file} {cfg_prefix}"
    cdk = "uv run cdk deploy"
    flags = f"--require-approval never --method=direct {rollback_flag} -c config_file={cfg_file}"
    return (
        "set -e\n"
        f"ALL=$({resolve} --phase 2)\n"
        f"STRATEGY=$({resolve} --strategy)\n"
        'if [ "$STRATEGY" = "combined" ]; then\n'
        f'  echo "Combined deploy (new or missing stacks)"\n'
        f"  {cdk} $ALL {flags}\n"
        "else\n"
        f"  P1=$({resolve} --phase 1)\n"
        f'  echo "Phase 1 - Deploying consumer stacks: $P1"\n'
        f"  {cdk} $P1 --exclusively {flags}\n"
        f'  echo "Phase 2 - Deploying all stacks: $ALL"\n'
        f"  {cdk} $ALL --exclusively {flags}\n"
        "fi\n"
        "echo 'All stacks deployed successfully'"
    )
