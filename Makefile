.PHONY: setup install-uv bootstrap lint test test-parallel test-marker \
       deploy deploy-manual synth view destroy clean

# ── One-time setup ───────────────────────────────────────────────────
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

bootstrap:
	cdk bootstrap

setup: install-uv
	uv venv --python 3.13
	uv sync --all-extras

# ── Quality gates ────────────────────────────────────────────────────
lint:
	uv run pre-commit run --all-files

test:
	uv run pytest tests/unit/ -x --no-header -q

test-parallel:
	uv run pytest tests/unit/ -x --no-header -q -n auto

# Usage: make test-marker MARKER=core
test-marker:
	uv run pytest tests/unit/ -m "$(MARKER)" -x --no-header -q

# ── Deploy via CI/CD (recommended) ──────────────────────────────────
# Runs lint → test → deploys all CI/CD stacks. Per-config source assets
# ensure only pipelines whose relevant files changed will trigger.
#
# Usage:
#   make deploy                                    # default cicd.yaml
#   make deploy CICD_CONFIG=cicd_prod.yaml         # alternate cicd config
CICD_FILE ?= $(or $(CICD_CONFIG),cicd.yaml)
SHARED_PREFIX := $(shell grep '^shared_prefix:' config/cicd/$(CICD_FILE) | awk '{print $$2}')

deploy: lint test
	cdk deploy $(SHARED_PREFIX)-SecurityStack $(SHARED_PREFIX)-CodeBuildStack $(SHARED_PREFIX)-CiCdPipelineStack $(SHARED_PREFIX)-ContainerPipelineStack \
		--require-approval never --no-rollback --output cdk.out \
		$(if $(CICD_CONFIG),-c cicd_config_file=$(CICD_CONFIG),)

# ── Manual deploy (bypasses CI/CD) ──────────────────────────────────
# Deploys all stacks for a single pipeline config directly.
# Defaults to config_vrag.yaml if CONFIG is not set.
#
# Usage:
#   make deploy-manual                             # default config_vrag.yaml
#   make deploy-manual CONFIG=config_t2i.yaml      # specific config
deploy-manual:
	cdk deploy --all --require-approval never --no-rollback \
		$(if $(CONFIG),-c config_file=$(CONFIG),)

# ── Synth ────────────────────────────────────────────────────────────
synth:
	cdk synth \
		$(if $(CONFIG),-c config_file=$(CONFIG),) \
		$(if $(CICD_CONFIG),-c cicd_config_file=$(CICD_CONFIG),)

# ── Streamlit viewer ────────────────────────────────────────────────
view:
	uv run streamlit run view_assets/app.py

# ── Destroy ──────────────────────────────────────────────────────────
# cdk destroy --all only tears down stacks synthesised in the current
# run.  app.py synths one pipeline config at a time, so a single
# invocation misses the DataStack / PipelineStack / A2IStack for the
# other configs.  Loop through every config in cicd.yaml so nothing
# is left behind.  Shared stacks (SecurityStack, CodeBuildStack,
# CiCdPipelineStack, ContainerPipelineStack) are covered by any run.
destroy:
	@for cfg in $$(awk '/^pipeline_configs:/{f=1;next} f&&/^  - /{gsub(/^  - /,"");gsub(/"/,"");print;next} f{exit}' config/cicd/cicd.yaml); do \
		echo "=== Destroying stacks for $$cfg ==="; \
		cdk destroy --all --force -c config_file=$$cfg --output cdk.out.destroy || true; \
	done
	@rm -rf cdk.out.destroy

# ── Clean build artifacts ────────────────────────────────────────────
clean:
	rm -rf cdk.out cdk.out.cicd .mypy_cache .pytest_cache .ruff_cache
