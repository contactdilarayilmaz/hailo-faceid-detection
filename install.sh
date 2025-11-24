#!/bin/bash
#
# Installation script for Hailo Face ID Detection
# This script sets up the Python environment and installs dependencies
#

set -e  # Exit on error

echo "🚀 Hailo Face ID Detection - Installation Script"
echo "================================================"

# Check if running on Raspberry Pi
if ! uname -a | grep -q "raspberrypi"; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi 5"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "📦 Python version: $PYTHON_VERSION"

# Check if virtual environment exists
VENV_NAME="venv_hailo_rpi_examples"
if [ -d "$VENV_NAME" ]; then
    echo "✅ Virtual environment already exists: $VENV_NAME"
else
    echo "📦 Creating virtual environment: $VENV_NAME"
    python3 -m venv $VENV_NAME
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing basic packages..."
    pip install numpy opencv-python annoy
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ .env file created. Please review and update if needed."
    else
        echo "⚠️  .env.example not found. Creating basic .env file..."
        cat > .env << EOF
host_arch=rpi
hailo_arch=hailo8
resources_path=resources
tappas_postproc_path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes

model_zoo_version=v2.14.0
hailort_version=4.20.0-1
tappas_version=3.31.0
virtual_env_name=venv_hailo_rpi_examples
server_url=http://dev-public.hailo.ai/2025_01
EOF
    fi
else
    echo "✅ .env file already exists"
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p data
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure Hailo SDK is installed on your system"
echo "   2. Activate the virtual environment: source $VENV_NAME/bin/activate"
echo "   3. Or use: source setup_env.sh"
echo "   4. Run the application: python3 -m basic_pipelines.detection_faceid --input rpi"
echo ""
echo "📖 For more information, see README.md"

