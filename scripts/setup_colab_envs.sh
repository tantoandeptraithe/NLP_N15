#!/usr/bin/env bash
# ============================================================================
# Setup Stage-Isolated Python 3.11 Virtual Environments on Google Colab using `uv`
# Usage:
#   bash scripts/setup_colab_envs.sh [all|stt|mt|tts_vieneu|tts_vixtts|eval_mt|eval_audio]
# ============================================================================
set -euo pipefail

TARGET="${1:-all}"
ENV_ROOT="${S2ST_ENV_ROOT:-/content/uv_envs}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv &> /dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$ENV_ROOT"

create_stage_env() {
    local name="$1"
    local req_file="$2"
    local venv_path="$ENV_ROOT/env-$name"

    echo "============================================================"
    echo "[setup] Building $venv_path (Python 3.11) from $req_file"
    echo "============================================================"
    uv venv "$venv_path" --python 3.11 --seed

    if [[ "$name" == "tts_vieneu" || "$name" == "tts_vixtts" || "$name" == "eval_audio" ]]; then
        uv pip install --python "$venv_path/bin/python" \
            torch==2.8.0 torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128
    fi

    uv pip install --python "$venv_path/bin/python" -r "$REPO_ROOT/requirements/$req_file"
    uv pip install --python "$venv_path/bin/python" -e "$REPO_ROOT" --no-deps

    # Lock exact resolved package versions for reproducibility
    "$venv_path/bin/python" -m pip freeze > "$REPO_ROOT/requirements/${name}.lock"
    echo "[setup] Locked -> requirements/${name}.lock"
}

case "$TARGET" in
    stt)        create_stage_env "stt" "stt.txt" ;;
    mt)         create_stage_env "mt" "mt.txt" ;;
    tts_vieneu) create_stage_env "tts_vieneu" "tts_vieneu.txt" ;;
    tts_vixtts) create_stage_env "tts_vixtts" "tts_vixtts.txt" ;;
    eval_mt)    create_stage_env "eval_mt" "eval_mt.txt" ;;
    eval_audio) create_stage_env "eval_audio" "eval_audio.txt" ;;
    all)
        create_stage_env "stt" "stt.txt"
        create_stage_env "mt" "mt.txt"
        create_stage_env "tts_vieneu" "tts_vieneu.txt"
        create_stage_env "tts_vixtts" "tts_vixtts.txt"
        create_stage_env "eval_mt" "eval_mt.txt"
        create_stage_env "eval_audio" "eval_audio.txt"
        ;;
    *)
        echo "Unknown target: $TARGET"
        exit 1
        ;;
esac
