FROM node:22-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
  git \
  curl \
  jq \
  && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && apt-get update && apt-get install -y gh \
  && rm -rf /var/lib/apt/lists/*

# Build-args for UID/GID alignment: sandcastle docker build-image
# defaults these to the host user's UID/GID so image-built files
# and bind-mounted files share an owner without runtime chown.
ARG AGENT_UID=1000
ARG AGENT_GID=1000

# Rename the base image's "node" user to "agent" and align UID/GID.
RUN groupmod -o -g $AGENT_GID node && usermod -o -u $AGENT_UID -g $AGENT_GID -d /home/agent -m -l agent node
USER ${AGENT_UID}:${AGENT_GID}

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Add Claude (and uv, installed below) to PATH
ENV PATH="/home/agent/.local/bin:$PATH"

# Install uv — the Python package/venv manager. uv also provisions the pinned
# Python toolchain (see .python-version / pyproject requires-python), so no
# system Python is needed. The implement/merge prompts run `make test` /
# `make typecheck`, which shell out to `uv run pytest` / `uv run mypy`.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Build the project venv at a container-only path OUTSIDE the bind-mounted
# worktree. Sandcastle bind-mounts the repo at /home/agent/workspace, so a
# default ./.venv would be written back onto the macOS host, where its
# Linux-aarch64 python symlinks are broken — and the VS Code Python and
# Black-formatter extensions then error on every refresh/format. Pointing uv
# at /home/agent/.venv keeps the venv inside the container and leaves the
# host .venv untouched.
ENV UV_PROJECT_ENVIRONMENT=/home/agent/.venv

WORKDIR /home/agent

# In worktree sandbox mode, Sandcastle bind-mounts the git worktree at /home/agent/workspace
# and overrides the working directory to /home/agent/workspace at container start.
# Structure your Dockerfile so that /home/agent/workspace can serve as the project root.
ENTRYPOINT ["sleep", "infinity"]
