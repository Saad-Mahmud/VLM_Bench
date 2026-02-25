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
                        - Use --cuda auto to pick a compatible runtime for this machine.
                        - If omitted with --profile vllm|all, defaults to --cuda auto.
                        - NOTE: nvidia-smi's "CUDA Version: X.Y" is a DRIVER version. It does not mean
                          pytorch-cuda=X.Y exists on conda. Driver CUDA 12.8 will typically map to a
                          supported pytorch-cuda runtime like 12.1.
  --recreate            Delete and recreate the env (DANGEROUS)
  -h, --help            Show this help

Examples:
  # Minimal (dataset pipeline)
  bash script/setup_server_env.sh --env nlp --profile data

  # vLLM server + eval (GPU; pick your CUDA version)
  bash script/setup_server_env.sh --env nlp --profile vllm --cuda 12.1

  # vLLM server + eval (auto-pick a compatible runtime)
  bash script/setup_server_env.sh --env nlp --profile vllm --cuda auto

  # A100 nodes often show "CUDA Version: 12.8" in nvidia-smi (driver),
  # which maps to a supported pytorch-cuda runtime like 12.1:
  bash script/setup_server_env.sh --env nlp --profile vllm --cuda 12.8
EOF
}

ENV_NAME="nlp"
PYTHON_VERSION="3.10"
PROFILE="data"
CUDA_VERSION=""
RECREATE=0

detect_driver_cuda_version() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi 2>/dev/null | awk -F 'CUDA Version: ' '/CUDA Version:/{split($2,a," "); print a[1]; exit}'
}

normalize_pytorch_cuda_version() {
  # Converts requested CUDA input (including driver versions like 12.8) into a conda pytorch-cuda runtime.
  # Goal: pick the "most compatible" pytorch-cuda runtime for vLLM/PyTorch binaries on this cluster.
  local requested="${1}"
  local driver_cuda="${2:-}"
  local default_cuda="12.1"

  if [[ -z "${requested}" ]]; then
    echo ""
    return 0
  fi

  if [[ "${requested}" == "auto" ]]; then
    if [[ -n "${driver_cuda}" ]]; then
      echo "Detected NVIDIA driver CUDA version: ${driver_cuda}" >&2
    else
      echo "No NVIDIA driver detected via nvidia-smi; using default pytorch-cuda runtime." >&2
    fi
    echo "${default_cuda}"
    return 0
  fi

  # Treat any 12.x other than 12.1 as a driver CUDA version and map to the supported runtime.
  if [[ "${requested}" =~ ^12\.[0-9]+$ ]] && [[ "${requested}" != "${default_cuda}" ]]; then
    echo "Mapping requested CUDA '${requested}' -> pytorch-cuda=${default_cuda} (driver CUDA != conda runtime)." >&2
    echo "${default_cuda}"
    return 0
  fi

  # Pass through (e.g., 12.1, 11.8).
  echo "${requested}"
}

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

if [[ -z "${CUDA_VERSION}" ]] && [[ "${PROFILE}" != "data" ]]; then
  CUDA_VERSION="auto"
fi

if [[ -n "${CUDA_VERSION}" ]]; then
  if [[ "${PROFILE}" == "data" ]]; then
    echo "ERROR: --cuda is only valid with --profile vllm|all" >&2
    exit 2
  fi

  DRIVER_CUDA_VERSION="$(detect_driver_cuda_version || true)"
  PYTORCH_CUDA_VERSION="$(normalize_pytorch_cuda_version "${CUDA_VERSION}" "${DRIVER_CUDA_VERSION}")"

  echo "Installing CUDA-enabled PyTorch via conda (pytorch-cuda=${PYTORCH_CUDA_VERSION})..."
  if ! conda install -y pytorch torchvision torchaudio "pytorch-cuda=${PYTORCH_CUDA_VERSION}" -c pytorch -c nvidia; then
    FALLBACK_CUDA_VERSION="11.8"
    if [[ "${PYTORCH_CUDA_VERSION}" != "${FALLBACK_CUDA_VERSION}" ]]; then
      echo "[WARN] PyTorch install failed for pytorch-cuda=${PYTORCH_CUDA_VERSION}. Retrying with pytorch-cuda=${FALLBACK_CUDA_VERSION}..."
      if ! conda install -y pytorch torchvision torchaudio "pytorch-cuda=${FALLBACK_CUDA_VERSION}" -c pytorch -c nvidia; then
        echo "ERROR: Failed to install CUDA-enabled PyTorch (tried pytorch-cuda=${PYTORCH_CUDA_VERSION} then ${FALLBACK_CUDA_VERSION})." >&2
        echo "       Try rerunning with: bash script/setup_server_env.sh --env ${ENV_NAME} --profile ${PROFILE} --cuda 12.1" >&2
        exit 1
      fi
      PYTORCH_CUDA_VERSION="${FALLBACK_CUDA_VERSION}"
      echo "[WARN] Installed PyTorch with fallback pytorch-cuda=${PYTORCH_CUDA_VERSION}."
    else
      echo "ERROR: Failed to install CUDA-enabled PyTorch (pytorch-cuda=${PYTORCH_CUDA_VERSION})." >&2
      echo "       Try rerunning with: bash script/setup_server_env.sh --env ${ENV_NAME} --profile ${PROFILE} --cuda 12.1" >&2
      exit 1
    fi
  fi

  python -c "import torch; print('torch', torch.__version__, 'torch_cuda', torch.version.cuda)" || true
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

if [[ "${PROFILE}" != "data" ]]; then
  python -c "import transformers, numpy as np; print('transformers', transformers.__version__, 'numpy', np.__version__)" || true
fi

echo
echo "Done."
echo "Activate env: conda activate ${ENV_NAME}"
