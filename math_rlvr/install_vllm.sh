#!/bin/bash

# vLLM Installation Script
# vLLM is a high-throughput and memory-efficient inference engine for LLMs

set -e

echo "=== vLLM Installation Script ==="

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "Found Python $PYTHON_VERSION"

        # vLLM requires Python 3.9+
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
            echo "Error: vLLM requires Python 3.9 or higher"
            exit 1
        fi
    else
        echo "Error: Python 3 is not installed"
        exit 1
    fi
}

# Check for CUDA
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        CUDA_AVAILABLE=true
    else
        echo "Warning: No NVIDIA GPU detected. vLLM works best with GPU support."
        CUDA_AVAILABLE=false
    fi
}

# Create virtual environment (optional)
setup_venv() {
    if [ "$1" == "--venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv vllm_env
        source vllm_env/bin/activate
        echo "Virtual environment activated"
    fi
}

# Install vLLM
install_vllm() {
    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing vLLM..."
    pip install vllm

    echo "vLLM installation complete!"
}

# Verify installation
verify_installation() {
    echo "Verifying installation..."
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" && \
        echo "Installation verified successfully!" || \
        echo "Warning: Could not verify installation"
}

# Main
main() {
    check_python
    check_cuda
    setup_venv "$1"
    install_vllm
    verify_installation

    echo ""
    echo "=== Installation Complete ==="
    echo ""
    echo "Quick start example:"
    echo "  from vllm import LLM, SamplingParams"
    echo "  llm = LLM(model='facebook/opt-125m')"
    echo "  outputs = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.8))"
    echo ""
    echo "For more info: https://docs.vllm.ai/"
}

main "$@"
