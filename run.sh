#!/usr/bin/env bash

set -euo pipefail

uv sync
uv run python download_model.py
uv run python pose_landmarker.py
