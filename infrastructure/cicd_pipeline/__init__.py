"""CI/CD pipeline stack, CodeBuild stack, and helper scripts."""

# Re-export everything from stack.py so that this package behaves like
# the old infrastructure.cicd_pipeline module. CDK internally resolves
# construct paths like infrastructure.cicd_pipeline.s3_assets.Asset,
# so the package namespace must expose the same attributes.
from infrastructure.cicd_pipeline.stack import *  # noqa: F401,F403
