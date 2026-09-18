#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEPP2_DIR="$ROOT/third_party/MEPP2"
MEPP2_COMMIT="d977d0543180b986132518d566dac78c2b8c9980"

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  echo "Missing cmake/ninja. Install system dependencies first:"
  echo "  sudo apt update"
  echo "  sudo apt install -y build-essential cmake ninja-build git libeigen3-dev libflann-dev liblz4-dev"
  exit 1
fi

if [[ ! -f /usr/include/eigen3/Eigen/Core ]] || [[ ! -f /usr/include/flann/flann.hpp ]]; then
  echo "Missing Eigen or FLANN headers. Install:"
  echo "  sudo apt install -y libeigen3-dev libflann-dev liblz4-dev"
  exit 1
fi

python -m pip install -U pybind11 numpy

mkdir -p "$ROOT/third_party"
if [[ ! -d "$MEPP2_DIR/.git" ]]; then
  echo "[1/4] Cloning only the required MEPP2 source subtree..."
  git clone --filter=blob:none --no-checkout https://github.com/MEPP-team/MEPP2.git "$MEPP2_DIR"
  git -C "$MEPP2_DIR" sparse-checkout init --cone
  git -C "$MEPP2_DIR" sparse-checkout set FEVV/Filters/Generic/PointCloud/WeightedPCANormals
else
  echo "[1/4] MEPP2 repository already exists."
fi

echo "[2/4] Pinning MEPP2 to commit $MEPP2_COMMIT"
git -C "$MEPP2_DIR" fetch --depth 1 origin "$MEPP2_COMMIT"
git -C "$MEPP2_DIR" checkout --detach FETCH_HEAD

echo "[3/4] Building wrapper..."
rm -rf "$ROOT/build"
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build "$ROOT/build" -j"$(nproc)"

echo "[4/4] Installing into the active Python environment..."
SITE_PACKAGES="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)"

MODULE_PATH="$(find "$ROOT/build" -maxdepth 2 -type f \( -name 'sanchez_wpca*.so' -o -name 'sanchez_wpca*.pyd' \) | head -n 1)"
if [[ -z "$MODULE_PATH" ]]; then
  echo "Built module not found."
  exit 1
fi

cp -f "$MODULE_PATH" "$SITE_PACKAGES/"

python - <<'PY'
import sanchez_wpca
print("sanchez_wpca import OK")
print(sanchez_wpca.__doc__.splitlines()[1])
PY

echo
echo "Installation complete."
echo "Test with: python $ROOT/examples/quick_test.py"
