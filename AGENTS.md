# Agent Instructions

## Default Conda Environment
- Environment name: `rc-llm-eval`
- Environment path: `/home/xuelin/miniconda3/envs/rc-llm-eval`
- Prefer running Python commands with `conda run -n rc-llm-eval ...` or `/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python`.

<!-- codex-agent-runtime:start -->

## Runtime Ports And Database Configuration

- Keep this section aligned with the root README when database names, ports, or service defaults change.
- Do not copy secrets from local `.env` files into commits; document only placeholders or compose defaults.

### Database
- No application database is used. Training and evaluation use local YAML configs, datasets, checkpoints, and result files.

### Default Ports
- No default web service or database port is defined.

### Notes For Codex Agents
- Use the Conda environment described in `environment.linux.yml` for experiments.
- Before committing, check `git status --short --branch` and avoid staging unrelated runtime artifacts.

### Source Files Checked
- `environment.linux.yml`
- `configs/*.yaml`
- `README.md`

<!-- codex-agent-runtime:end -->

## GitHub Commit Language

- Use English for all GitHub commit messages and pull/push related commit notes.
