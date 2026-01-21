#!/bin/bash

# Спираме скрипта при всяка грешка (fail fast)
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"
REQ_FILE="$BACKEND_DIR/requirements.txt"
MODEL_DIR="$BACKEND_DIR/models"

echo "🚀 Starting Environment Setup for Jarvis (M4 Pro Edition)..."

# 1. System Dependencies Check (Crucial for sounddevice)
echo "🔍 Checking system dependencies..."

if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first!"
    exit 1
fi

if ! brew list portaudio &> /dev/null; then
    echo "📦 PortAudio not found. Installing via Homebrew (needed for Microphone)..."
    brew install portaudio
else
    echo "✅ PortAudio is already installed."
fi

# 2. Python Environment Setup
# Проверяваме дали имаме стара среда и я зачистваме ако трябва (по желание)
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Found existing venv. Activating it..."
else
    echo "🔨 Creating new Python virtual environment (venv)..."
    python3 -m venv "$VENV_DIR"
fi

# 3. Activation & Upgrade
echo "🔌 Activating venv..."
source "$VENV_DIR/bin/activate"

echo "⬆️  Upgrading pip to latest version..."
pip install --upgrade pip

# 4. Install Dependencies
if [ -f "$REQ_FILE" ]; then
    echo "📥 Installing libraries from requirements.txt (This might take a moment)..."
    pip install -r "$REQ_FILE"
else
    echo "❌ requirements.txt not found! Create it first."
    exit 1
fi

# 5. Download Kokoro Model (Optional convenience)
# Спестяваме ти търсенето на файла. Дърпаме v0.19 ONNX модела.
mkdir -p "$MODEL_DIR"
KOKORO_FILE="$MODEL_DIR/kokoro-v0_19.onnx"
VOICES_FILE="$MODEL_DIR/voices.bin"

if [ ! -f "$KOKORO_FILE" ]; then
    echo "⬇️  Downloading Kokoro ONNX model (~80MB)..."
    curl -L -o $KOKORO_FILE "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
fi

if [ ! -f "$VOICES_FILE" ]; then
    echo "⬇️  Downloading Kokoro Voices config..."
    curl -L -o $VOICES_FILE "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
fi

echo "✅ Setup Complete!"
echo "👉 To start working, run: source backend/venv/bin/activate"
