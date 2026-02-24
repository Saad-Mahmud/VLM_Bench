#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash script/setup_server_env.sh [options]

Creates/uses a conda env, installs `uv`, then installs the Python deps needed for this repo.

Options:
  --env NAME            Conda env name (default: nlp)
  --python VERSION      Python version for new env (default: 3.10)
  --profile PROFILE     One of: data | vllm | all (default: data)
  --cuda VERSION        (vllm/all) Install CUDA-enabled PyTorch via conda using pytorch-cuda=VERSION
  --recreate            Delete and recreate the env (DANGEROUS)
  -h, --help            Show this help

Examples:
  # Minimal (dataset pipeline)
  bash script/setup_server_env.sh --env nlp --profile data

  # vLLM server + eval (GPU; pick your CUDA version)
  bash script/setup_server_env.sh --env nlp --profile vllm --cuda 12.1
EOF
}

ENV_NAME="nlp"
PYTHON_VERSION="3.10"
PROFILE="data"
CUDA_VERSION=""
RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env-name)
      ENV_NAME="${2:?missing value for --env}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift 2
      ;;
    --cuda)
      CUDA_VERSION="${2:?missing value for --cuda}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Install Miniconda/Anaconda and re-run." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ "${RECREATE}" -eq 1 ]]; then
  echo "[WARN] Recreating conda env: ${ENV_NAME}"
  conda env remove -y -n "${ENV_NAME}" >/dev/null 2>&1 || true
fi

if ! conda env list | awk '{print $1}' | grep -Ev '^(#|$)' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel
python -m pip install -U uv

PKGS_DATA=(
  datasets
  pillow
)

PKGS_VLLM=(
  fastapi
  "uvicorn[standard]"
  pydantic
  transformers
  huggingface_hub
  mistral-common
  vllm
)

PKGS_ALL=(
  "${PKGS_DATA[@]}"
  "${PKGS_VLLM[@]}"
  matplotlib
)

case "${PROFILE}" in
  data)
    PKGS=("${PKGS_DATA[@]}")
    ;;
  vllm)
    PKGS=("${PKGS_DATA[@]}" "${PKGS_VLLM[@]}")
    ;;
  all)
    PKGS=("${PKGS_ALL[@]}")
    ;;
  *)
    echo "ERROR: Unknown --profile '${PROFILE}'. Expected: data | vllm | all" >&2
    exit 2
    ;;
esac

if [[ -n "${CUDA_VERSION}" ]]; then
  if [[ "${PROFILE}" == "data" ]]; then
    echo "ERROR: --cuda is only valid with --profile vllm|all" >&2
    exit 2
  fi
  echo "Installing CUDA-enabled PyTorch via conda (pytorch-cuda=${CUDA_VERSION})..."
  conda install -y pytorch torchvision torchaudio "pytorch-cuda=${CUDA_VERSION}" -c pytorch -c nvidia
else
  if [[ "${PROFILE}" != "data" ]]; then
    cat <<'EOF'
[NOTE] vLLM needs a CUDA-enabled PyTorch build.
       If your server has NVIDIA GPUs, re-run with e.g. `--cuda 12.1` (or install PyTorch+CUDA separately).
EOF
  fi
fi

echo "Installing Python packages via uv into conda env '${ENV_NAME}'..."
uv pip install --python "$(command -v python)" "${PKGS[@]}"

echo
echo "Done."
echo "Activate env: conda activate ${ENV_NAME}"
