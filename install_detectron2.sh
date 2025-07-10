#!/usr/bin/env bash
# Detectron2 + PyTorch 2.4.1 (CUDA 12.1) installer  –  2025‑07‑08 v5
# Works on Ubuntu 22.04 desktops and AWS G5 (A10 G)

set -euo pipefail

########################  CONFIG  ########################
CUDA_PKG="12-1"            # dash  form → apt package: cuda-toolkit-12-1
CUDA_DIR="12.1"            # dotted form → actual dir: /usr/local/cuda-12.1
DRIVER_FALLBACK="550"      # GA driver branch in Ubuntu 22.04 repos
VENV_NAME="d2"             # virtual‑env will be ~/envs/d2
PYTORCH_VERSION="2.4.1"
TORCHVISION_VERSION="0.19.1"
CUDA_ARCH="8.6"            # A10 G compute capability
##########################################################

green() { printf '\e[1;32m%s\e[0m\n' "$*"; }
die()   { printf '\e[1;31m%s\e[0m\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] && die "Run as a normal user, not root."

##########################################################
# 1) NVIDIA DRIVER
##########################################################
if ! command -v nvidia-smi >/dev/null; then
  green "→ Installing NVIDIA driver …"
  sudo apt update
  sudo apt install -y ubuntu-drivers-common
  if command -v ubuntu-drivers >/dev/null; then
    sudo ubuntu-drivers autoinstall
  else
    sudo apt install -y "nvidia-driver-${DRIVER_FALLBACK}"
  fi
  green "→ Reboot required so the driver loads.  Re‑run this script afterwards."
  exit 0
else
  green "✓ NVIDIA driver present ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader))"
fi

##########################################################
# 2) CUDA 12.1 TOOLKIT  (ADD REPO IF NEEDED)
##########################################################
if ! { command -v nvcc >/dev/null && nvcc --version | grep -q "release ${CUDA_DIR}"; }; then
  green "→ Installing CUDA toolkit ${CUDA_DIR} …"
  if ! apt-cache show "cuda-toolkit-${CUDA_PKG}" >/dev/null 2>&1; then
    green "→ Adding NVIDIA CUDA apt repository …"
    sudo apt install -y wget gnupg
    wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt update
  fi
  sudo apt install -y "cuda-toolkit-${CUDA_PKG}"
else
  green "✓ CUDA toolkit ${CUDA_DIR} already installed"
fi

# --- 2b)  EXPORT CUDA ${CUDA_DIR} PATHS (CURRENT SHELL + FUTURE) -----
CUDA_HOME="/usr/local/cuda-${CUDA_DIR}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found in ${CUDA_HOME}; toolkit install failed."

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

grep -q "cuda-${CUDA_DIR}/bin" ~/.bashrc || {
  green "→ Adding CUDA paths to ~/.bashrc …"
  {
    echo "export PATH=${CUDA_HOME}/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH"
  } >> ~/.bashrc
}

##########################################################
# 3) BUILD TOOLS & PYTHON HEADERS
##########################################################
green "→ Installing build dependencies …"
sudo apt install -y build-essential gcc-11 g++-11 \
                    git cmake ninja-build libjpeg-dev libpng-dev \
                    python3.10 python3.10-venv python3-dev

##########################################################
# 4) PYTHON VENV
##########################################################
VENV_HOME="$HOME/envs/${VENV_NAME}"
if [[ ! -d "$VENV_HOME" ]]; then
  green "→ Creating virtual‑env ${VENV_HOME} …"
  python3.10 -m venv "$VENV_HOME"
fi
# shellcheck disable=SC1090
source "${VENV_HOME}/bin/activate"
pip install --quiet --upgrade pip

##########################################################
# 5) PyTorch 2.4.1 + CUDA 12.1
##########################################################
green "→ Installing PyTorch ${PYTORCH_VERSION} (cu121) …"
pip install --quiet \
  "torch==${PYTORCH_VERSION}+cu121" \
  "torchvision==${TORCHVISION_VERSION}+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

##########################################################
# 6) wheel + setuptools + cython
##########################################################
green "→ Ensuring wheel / setuptools / cython …"
pip install --quiet --upgrade wheel setuptools cython

##########################################################
# 7) DETECTRON2  (no‑build‑isolation)
##########################################################
green "→ Building Detectron2 …"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
pip install --quiet --no-build-isolation --no-cache-dir \
  'git+https://github.com/facebookresearch/detectron2.git'

##########################################################
# 8) SMOKE TEST
##########################################################
python - <<'PY'
import torch, detectron2, os
print("\n✅  Install complete")
print("    Detectron2:", detectron2.__version__)
print("    PyTorch:   ", torch.__version__, "CUDA", torch.version.cuda)
print("    GPU:       ", torch.cuda.get_device_name(0))
PY

green "\n🎉 Detectron2 is ready!"
echo "   Activate it anytime with:"
echo "   source ${VENV_HOME}/bin/activate"
