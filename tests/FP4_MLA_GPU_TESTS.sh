#!/usr/bin/env bash
set -euo pipefail

# Deferred GPU smoke tests for nvfp4_ds_mla.
# Do not run this until /home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN exists.
# This script intentionally avoids protected containers and ports 9200/9300.

ROOT="${ROOT:-/home/brandonmusic/klc-linux}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/fp4_mla_build}"
GATE="${GATE:-${ROOT}/FP4_GPU_WINDOW_OPEN}"
IMAGE_TAG="${IMAGE_TAG:-klc/fp4mla-test:dev}"
BASE_IMAGE="${BASE_IMAGE:-voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618}"

MODEL_DIR="${MODEL_DIR:-/home/brandonmusic/models/GLM-5.2-NVFP4}"
MODEL_PATH="${MODEL_PATH:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

FP8_NAME="${FP8_NAME:-fp4mla-fp8-baseline}"
NVFP4_NAME="${NVFP4_NAME:-fp4mla-nvfp4-test}"
FP8_PORT="${FP8_PORT:-9401}"
NVFP4_PORT="${NVFP4_PORT:-9402}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

OUTDIR="${OUTDIR:-${BUILD_ROOT}/gpu_results/$(date -u +%Y%m%dT%H%M%SZ)}"
BENCH="${BENCH:-/home/brandonmusic/llm-inference-bench/llm_decode_bench.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_gpu_window() {
  [[ -e "${GATE}" ]] || die "GPU gate file is absent: ${GATE}"
}

reject_protected_identity() {
  local value="$1"
  case "${value}" in
    glm52-9300-test|dsv4-9200-prod)
      die "refusing protected container name: ${value}"
      ;;
  esac
}

reject_protected_port() {
  local value="$1"
  case "${value}" in
    9200|9300)
      die "refusing protected port: ${value}"
      ;;
  esac
}

preflight() {
  require_gpu_window
  reject_protected_identity "${FP8_NAME}"
  reject_protected_identity "${NVFP4_NAME}"
  reject_protected_port "${FP8_PORT}"
  reject_protected_port "${NVFP4_PORT}"
  [[ "${FP8_PORT}" != "${NVFP4_PORT}" ]] || die "FP8_PORT and NVFP4_PORT must differ"
  [[ -f "${BENCH}" ]] || die "benchmark script not found: ${BENCH}"
  [[ -d "${MODEL_DIR}" ]] || die "host MODEL_DIR not found: ${MODEL_DIR}"
  mkdir -p "${OUTDIR}"
}

build_preflight() {
  [[ -f "${BUILD_ROOT}/docker_context/Dockerfile" ]] || die "Dockerfile missing under ${BUILD_ROOT}/docker_context"
}

build_image() {
  build_preflight
  docker build \
    --pull=false \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${BUILD_ROOT}/docker_context"
}

start_server() {
  local name="$1"
  local host_port="$2"
  local kv_dtype="$3"

  reject_protected_identity "${name}"
  reject_protected_port "${host_port}"

  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${name}" \
    --gpus "device=${GPU_IDS}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "127.0.0.1:${host_port}:${CONTAINER_PORT}" \
    -v "${MODEL_DIR}:/model:ro" \
    -v "/home/brandonmusic/.cache/huggingface:/root/.cache/huggingface:rw" \
    -v "/home/brandonmusic/.cache/fp4mla-triton:/root/.cache/triton:rw" \
    -e "MODEL=${MODEL_PATH}" \
    -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
    -e "PORT=${CONTAINER_PORT}" \
    -e "TP_SIZE=${TP_SIZE}" \
    -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
    -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
    -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
    -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    -e "ATTENTION_BACKEND=B12X_MLA_SPARSE" \
    -e "KV_CACHE_DTYPE=${kv_dtype}" \
    -e "CUTE_DSL_ARCH=sm_120" \
    -e "TORCH_CUDA_ARCH_LIST=12.0" \
    "${IMAGE_TAG}" \
    bash -lc 'cd /opt/vllm && exec ./serve-glm52.sh'
}

wait_ready() {
  local name="$1"
  local port="$2"
  local log_file="${OUTDIR}/${name}.startup.log"

  : > "${log_file}"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      docker logs "${name}" > "${log_file}" 2>&1 || true
      return 0
    fi
    local status
    status="$(docker ps -a --filter "name=^/${name}$" --format '{{.Status}}' || true)"
    case "${status}" in
      Exited*|Dead*)
        docker logs "${name}" > "${log_file}" 2>&1 || true
        die "${name} exited before readiness; see ${log_file}"
        ;;
    esac
    sleep 5
  done
  docker logs "${name}" > "${log_file}" 2>&1 || true
  die "${name} did not become ready; see ${log_file}"
}

start_pair() {
  preflight
  build_image
  start_server "${FP8_NAME}" "${FP8_PORT}" "fp8_ds_mla"
  wait_ready "${FP8_NAME}" "${FP8_PORT}"
  start_server "${NVFP4_NAME}" "${NVFP4_PORT}" "nvfp4_ds_mla"
  wait_ready "${NVFP4_NAME}" "${NVFP4_PORT}"
}

stop_pair() {
  reject_protected_identity "${FP8_NAME}"
  reject_protected_identity "${NVFP4_NAME}"
  docker rm -f "${FP8_NAME}" "${NVFP4_NAME}" >/dev/null 2>&1 || true
}

decode_equivalence() {
  preflight
  "${PYTHON_BIN}" - "${FP8_PORT}" "${NVFP4_PORT}" "${SERVED_MODEL_NAME}" "${OUTDIR}/decode_equivalence.json" <<'PY'
import json
import math
import sys
import urllib.request

fp8_port, nvfp4_port, model, out_path = sys.argv[1:5]
prompt = (
    "You are comparing two KV cache formats. Answer with exactly one short "
    "paragraph: explain why RoPE channels should stay high precision in long "
    "context MLA retrieval."
)

def request(port: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 96,
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "logprobs": -1,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))

def tokens(resp: dict) -> list[str]:
    choice = resp["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return list(logprobs.get("tokens") or [])

def top_logprobs(resp: dict) -> list[dict[str, float]]:
    choice = resp["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return list(logprobs.get("top_logprobs") or [])

def sparse_cosine(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    av = {k: math.exp(float(a.get(k, -100.0))) for k in keys}
    bv = {k: math.exp(float(b.get(k, -100.0))) for k in keys}
    dot = sum(av[k] * bv[k] for k in keys)
    na = math.sqrt(sum(v * v for v in av.values()))
    nb = math.sqrt(sum(v * v for v in bv.values()))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)

fp8 = request(fp8_port)
nvfp4 = request(nvfp4_port)
fp8_tokens = tokens(fp8)
nvfp4_tokens = tokens(nvfp4)
divergence = None
for i, (a, b) in enumerate(zip(fp8_tokens, nvfp4_tokens)):
    if a != b:
        divergence = i
        break
if divergence is None and len(fp8_tokens) != len(nvfp4_tokens):
    divergence = min(len(fp8_tokens), len(nvfp4_tokens))

cosines = [
    sparse_cosine(a or {}, b or {})
    for a, b in zip(top_logprobs(fp8), top_logprobs(nvfp4))
]
cosines = [c for c in cosines if c is not None]
summary = {
    "prompt": prompt,
    "fp8_port": int(fp8_port),
    "nvfp4_port": int(nvfp4_port),
    "divergence_token_index": divergence,
    "top_logprob_cosine_min": min(cosines) if cosines else None,
    "top_logprob_cosine_mean": sum(cosines) / len(cosines) if cosines else None,
    "fp8_text": fp8["choices"][0].get("text", ""),
    "nvfp4_text": nvfp4["choices"][0].get("text", ""),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_profile_pair() {
  local profile="$1"
  local runs="$2"
  local max_tokens="$3"
  preflight

  for side in fp8 nvfp4; do
    local port
    case "${side}" in
      fp8) port="${FP8_PORT}" ;;
      nvfp4) port="${NVFP4_PORT}" ;;
    esac
    "${PYTHON_BIN}" "${BENCH}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --model "${SERVED_MODEL_NAME}" \
      --test-profile "${profile}" \
      --completion-stats-concurrency 1 \
      --completion-stats-runs "${runs}" \
      --completion-stats-temperature 1.0 \
      --completion-stats-top-p 0.95 \
      --completion-stats-score-source final_answer \
      --max-tokens "${max_tokens}" \
      --display-mode plain \
      --no-hw-monitor \
      --output "${OUTDIR}/${side}_${profile}.json" \
      2>&1 | tee "${OUTDIR}/${side}_${profile}.log"
  done
}

retrieval_gate() {
  run_profile_pair estonia 10 8192
  run_profile_pair lavd-test 5 32768
  run_profile_pair hotel-lights 10 4096
}

ruler_gate() {
  preflight
  if [[ -z "${RULER_CMD:-}" ]]; then
    cat >&2 <<'EOF'
Set RULER_CMD to the local RULER command before running this step.
It must emit machine-readable results for 32K, 64K, 128K, and 256K for both ports.
Required gate: nvfp4_ds_mla has no more than a 1-2 point drop vs fp8_ds_mla at 64K-128K.
Example shape:
  RULER_CMD='python3 /path/to/ruler_runner.py --host 127.0.0.1 --model GLM-5.2 --contexts 32k,64k,128k,256k'
EOF
    exit 2
  fi
  FP8_PORT="${FP8_PORT}" NVFP4_PORT="${NVFP4_PORT}" OUTDIR="${OUTDIR}" bash -lc "${RULER_CMD}"
}

capacity_speed() {
  preflight
  for side in fp8 nvfp4; do
    local port
    case "${side}" in
      fp8) port="${FP8_PORT}" ;;
      nvfp4) port="${NVFP4_PORT}" ;;
    esac
    "${PYTHON_BIN}" "${BENCH}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --model "${SERVED_MODEL_NAME}" \
      --contexts 32k,64k,128k \
      --concurrency 1,2,4,8 \
      --duration 30 \
      --max-tokens 512 \
      --completion-stats-temperature 1.0 \
      --completion-stats-top-p 0.95 \
      --display-mode plain \
      --no-hw-monitor \
      --output "${OUTDIR}/${side}_capacity_speed.json" \
      2>&1 | tee "${OUTDIR}/${side}_capacity_speed.log"
  done
}

case "${1:-help}" in
  build)
    build_image
    ;;
  start)
    start_pair
    ;;
  stop)
    stop_pair
    ;;
  decode)
    decode_equivalence
    ;;
  retrieval)
    retrieval_gate
    ;;
  ruler)
    ruler_gate
    ;;
  speed)
    capacity_speed
    ;;
  all)
    start_pair
    decode_equivalence
    retrieval_gate
    ruler_gate
    capacity_speed
    ;;
  help|--help|-h)
    cat <<EOF
Usage: $0 {build|start|stop|decode|retrieval|ruler|speed|all}

Defaults:
  IMAGE_TAG=${IMAGE_TAG}
  FP8 endpoint:    ${FP8_NAME} on 127.0.0.1:${FP8_PORT}
  NVFP4 endpoint:  ${NVFP4_NAME} on 127.0.0.1:${NVFP4_PORT}
  Output dir:      ${OUTDIR}

This script exits unless ${GATE} exists.
It never uses container names glm52-9300-test/dsv4-9200-prod or ports 9200/9300.
EOF
    ;;
  *)
    die "unknown command: $1"
    ;;
esac
