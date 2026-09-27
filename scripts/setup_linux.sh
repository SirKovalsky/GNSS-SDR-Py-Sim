#!/usr/bin/env bash
# Standalone setup for GNSS Sim on Linux / macOS.
#
# Creates a project-local virtual environment (.venv), installs the Python
# requirements and, on Debian/Ubuntu with the UHD packages available, installs
# UHD via apt (with --system-site-packages so `import uhd` resolves) and
# downloads the FPGA images.  Idempotent: safe to re-run.
#
# Usage:  ./scripts/setup_linux.sh
#         PYTHON=python3.13 ./scripts/setup_linux.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

echo "== GNSS Sim setup =="
echo "Project: $ROOT"

# --- 1. UHD via apt (best effort; needs sudo + network) --------------------
UHD_APT=0
if command -v apt-get >/dev/null 2>&1 && apt-cache show libuhd-dev >/dev/null 2>&1; then
    echo "UHD packages available - installing libuhd-dev uhd-host python3-uhd"
    if sudo apt-get install -y libuhd-dev uhd-host python3-uhd; then
        UHD_APT=1
    else
        echo "WARNING: apt install of UHD failed; continuing without it." >&2
    fi
else
    echo "UHD apt packages not found (non-Debian system or no network)."
fi

# --- 2. Project-local virtual environment ----------------------------------
# UHD from apt lives in the system site-packages, so the venv is created with
# --system-site-packages to keep `import uhd` working.  Recreate an existing
# venv that cannot see the system uhd.
if [ "$UHD_APT" = "1" ]; then
    if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import uhd' >/dev/null 2>&1; then
        echo "Recreating .venv with --system-site-packages (for apt UHD)"
        rm -rf .venv
    fi
    if [ ! -x .venv/bin/python ]; then
        "$PYTHON" -m venv --system-site-packages .venv
    fi
else
    if [ ! -x .venv/bin/python ]; then
        "$PYTHON" -m venv .venv
    fi
fi

echo "Upgrading pip"
.venv/bin/python -m pip install --upgrade pip
echo "Installing requirements.txt"
.venv/bin/python -m pip install -r requirements.txt

# --- 3. UHD images / Python bindings ---------------------------------------
if [ "$UHD_APT" = "1" ]; then
    if command -v uhd_images_downloader >/dev/null 2>&1; then
        echo "Downloading UHD FPGA images"
        sudo uhd_images_downloader || echo "WARNING: uhd_images_downloader failed" >&2
    fi
    if ! .venv/bin/python -c 'import uhd' >/dev/null 2>&1; then
        echo "System uhd not importable - trying pip uhd==4.10.0.0"
        .venv/bin/python -m pip install "uhd==4.10.0.0" \
            || echo "WARNING: pip uhd install failed; file-only mode still works." >&2
    fi
else
    cat <<'EOF'
No system UHD installed.  For USRP B210 TX either:
  * Debian/Ubuntu: sudo apt install libuhd-dev uhd-host python3-uhd && ./scripts/setup_linux.sh
  * or build UHD from source with -DENABLE_PYTHON_API=ON, then:
        .venv/bin/python -m pip install uhd==4.10.0.0
Without UHD the simulator still generates IQ files.
EOF
fi

chmod +x "$ROOT/scripts/setup_linux.sh" 2>/dev/null || true

echo ""
echo "Setup complete."
echo "Run the GUI : .venv/bin/python run.py"
echo "Run the CLI : .venv/bin/python -m gnss_sim --help"
echo "Run tests   : .venv/bin/python -m pytest tests -q"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected - GPU synthesis: .venv/bin/python -m pip install 'cupy-cuda12x<14'"
fi
