#!/bin/bash

# AnomalyCLIP Dashboard Startup Script
# This script starts the FastAPI server with proper environment configuration

set -e

echo "🚀 Starting AnomalyCLIP Dashboard..."
echo ""

# Navigate to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)
echo "📁 Project root: $PROJECT_ROOT"

# Check for checkpoints
CHECKPOINT_COUNT=$(find checkpoints -name "*.ckpt" 2>/dev/null | wc -l | tr -d ' ')
if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: No checkpoint files found in checkpoints/"
    echo "   Please download a checkpoint and place it in:"
    echo "   - checkpoints/ucfcrime/last.ckpt (preferred)"
    echo "   - checkpoints/shanghaitech/last.ckpt"
    echo ""
fi

# Check for labels
LABELS_COUNT=$(find data -name "*labels*.csv" 2>/dev/null | wc -l | tr -d ' ')
if [ "$LABELS_COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: No labels CSV found in data/"
    echo "   Please add a labels file:"
    echo "   - data/ucf_labels.csv"
    echo "   - data/sht_labels.csv"
    echo ""
fi

# Detect Python command (conda env uses 'python', system uses 'python3')
if command -v python &> /dev/null && python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.8+ not found"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Check PyTorch and MPS
echo "🔍 Checking PyTorch installation..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not found. Please install: pip install torch torchvision"
    exit 1
}

echo "🔍 Checking MPS (Metal Performance Shaders) support..."
MPS_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "false")
if [ "$MPS_AVAILABLE" = "True" ]; then
    echo "✅ MPS available - will use GPU acceleration"
    export ANOMALYCLIP_DEVICE=mps
else
    echo "ℹ️  MPS not available - will use CPU"
    export ANOMALYCLIP_DEVICE=cpu
fi

# Set project root environment variable
export PROJECT_ROOT="$PROJECT_ROOT"

# Configuration
export SERVE_INFER_INTERVAL=4
export SERVE_MAX_EMIT_FPS=8.0
export SERVE_MAX_WIDTH=1280

echo ""
echo "📋 Configuration:"
echo "   Device: $ANOMALYCLIP_DEVICE"
echo "   Inference interval: $SERVE_INFER_INTERVAL frames"
echo "   Max emit FPS: $SERVE_MAX_EMIT_FPS"
echo "   Max frame width: $SERVE_MAX_WIDTH px"
echo ""

# Check if Ollama is running (optional)
echo "🔍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running - AI summaries will be available"
    OLLAMA_MODELS=$(curl -s http://localhost:11434/api/tags | $PYTHON_CMD -c "import sys, json; models = json.load(sys.stdin).get('models', []); print(', '.join([m['name'] for m in models]))" 2>/dev/null || echo "")
    if [ ! -z "$OLLAMA_MODELS" ]; then
        echo "   Available models: $OLLAMA_MODELS"
    fi
else
    echo "ℹ️  Ollama not running - AI summaries will be disabled"
    echo "   To enable: brew install ollama && ollama serve && ollama pull minicpm-v"
fi

echo ""
echo "🌐 Starting FastAPI server..."
echo "   URL: http://localhost:8000"
echo "   Dashboard: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server
cd "$PROJECT_ROOT"
$PYTHON_CMD -m uvicorn src.server.main:app --reload --host 0.0.0.0 --port 8000
