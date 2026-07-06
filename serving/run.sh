#!/usr/bin/env bash
# GLM-5.2-MXFP8-NVFP4-NF3-Hybrid + nvfp4_ds_mla 4-bit MLA KV cache — copy-paste run.
# 4x RTX PRO 6000 96GB (SM120), TP4/DCP4. Everything (env + launch) is baked into the image.
# Only set MODEL_DIR to your downloaded checkpoint dir.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=/path/to/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
docker run --rm --name glm52-nvfp4-kv \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR:/model:ro" -v glm52-nvfp4-cache:/cache \
  verdictai/glm52-nvfp4-kv:v2
