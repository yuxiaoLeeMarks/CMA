#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 EXP_NAME IN_DATASET [CLIP_CKPT] [DATA_ROOT] [AGENT_RATIO] [AGENT_FILE]" >&2
  exit 1
fi

EXP_NAME=$1
ID=$2
CKPT=${3:-ViT-B/16}
DATA_ROOT=${4:-datasets}
AGENT_RATIO=${5:-1.0}
AGENT_FILE=${6:-}

CMD=(python eval_ood_detection.py
  --in_dataset "${ID}"
  --name "${EXP_NAME}"
  --CLIP_ckpt "${CKPT}"
  --score CMA
  --root-dir "${DATA_ROOT}"
  --agent-ratio "${AGENT_RATIO}"
)

if [ -n "${AGENT_FILE}" ]; then
  CMD+=(--agent-prompts-file "${AGENT_FILE}")
fi

"${CMD[@]}"
