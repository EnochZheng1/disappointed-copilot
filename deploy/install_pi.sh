#!/bin/bash
# Disappointed Co-Pilot — Raspberry Pi 5 Setup Script
# Run on a fresh Raspberry Pi OS (Bookworm, 64-bit)
#
# Usage: sudo bash deploy/install_pi.sh
set -euo pipefail

INSTALL_DIR="/opt/disappointed"
VENV_DIR="$INSTALL_DIR/venv"
USER="${SUDO_USER:-pi}"

echo "=== Disappointed Co-Pilot — Pi 5 Setup ==="

# --- System dependencies ---
echo "[1/7] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-pip python3-venv python3-dev \
    libcap-dev libatlas-base-dev \
    libopencv-dev python3-opencv \
    libcamera-dev python3-libcamera python3-picamera2 \
    ffmpeg portaudio19-dev \
    curl git

# --- Google Coral EdgeTPU runtime ---
echo "[2/7] Installing Coral EdgeTPU runtime..."
if ! dpkg -l | grep -q libedgetpu1-std; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > \
        /etc/apt/sources.list.d/coral-edgetpu.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt-get update
    apt-get install -y libedgetpu1-std python3-pycoral
fi

# --- Active cooling setup ---
echo "[3/7] Configuring active cooling..."
# Enable fan control via device tree overlay
if ! grep -q "dtoverlay=gpio-fan" /boot/firmware/config.txt 2>/dev/null; then
    echo "" >> /boot/firmware/config.txt
    echo "# Disappointed Co-Pilot — Active cooling" >> /boot/firmware/config.txt
    echo "dtoverlay=gpio-fan,gpiopin=14,temp=60000" >> /boot/firmware/config.txt
    echo "  Fan GPIO overlay added (activates at 60°C)"
fi

# --- Install project ---
echo "[4/7] Installing Disappointed Co-Pilot..."
mkdir -p "$INSTALL_DIR"
if [ -d "$INSTALL_DIR/src" ]; then
    echo "  Existing installation found, updating..."
else
    cp -r . "$INSTALL_DIR/"
fi

python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -e "$INSTALL_DIR[pi]"

# --- Download YOLO model for EdgeTPU ---
echo "[5/7] Model setup..."
mkdir -p "$INSTALL_DIR/models"
echo "  NOTE: You need to export and copy your YOLOv8n EdgeTPU model to:"
echo "  $INSTALL_DIR/models/yolov8n_full_integer_quant_edgetpu.tflite"
echo "  See scripts/export_model.py for instructions."

# --- Install Ollama (optional) ---
echo "[6/7] Installing Ollama (optional LLM engine)..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Pulling llama3.2:1b model..."
    sudo -u "$USER" ollama pull llama3.2:1b || echo "  Warning: Ollama model pull failed (can retry later)"
fi

# --- Systemd service ---
echo "[7/7] Setting up systemd service..."
cp "$INSTALL_DIR/deploy/disappointed.service" /etc/systemd/system/
# Update service file with correct user
sed -i "s/User=pi/User=$USER/" /etc/systemd/system/disappointed.service
systemctl daemon-reload
systemctl enable disappointed

# --- Create clips output directory ---
mkdir -p /media/usb/clips 2>/dev/null || mkdir -p "$INSTALL_DIR/clips"
chown -R "$USER:$USER" "$INSTALL_DIR"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Hardware checklist:"
echo "  [ ] Pi 5 Active Cooler installed (MANDATORY)"
echo "  [ ] White/reflective case with ventilation"
echo "  [ ] 5V/5A hardwired car power supply"
echo "  [ ] Pi Camera Module 3 (wide-angle) connected"
echo "  [ ] Google Coral USB Accelerator plugged in"
echo "  [ ] EdgeTPU model exported to $INSTALL_DIR/models/"
echo ""
echo "To start manually:  sudo systemctl start disappointed"
echo "To view logs:        journalctl -u disappointed -f"
echo "To stop:             sudo systemctl stop disappointed"
echo ""
echo "NOTE: Reboot recommended to enable fan GPIO overlay."
