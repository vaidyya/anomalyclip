#!/bin/bash

# Optimization Script for Short Videos (< 30 seconds)
# This adjusts the temporal window to match short video lengths

echo "🎯 Optimizing AnomalyCLIP for Short Videos"
echo "=========================================="
echo ""

# Original UCF-Crime settings (for long surveillance videos):
# - num_segments = 32
# - seg_length = 16
# - stride = 16
# - window = 512 frames (~21 seconds at 24fps)

# Optimized settings for short videos (10-30 seconds):
# - Reduce window size to match video length
# - Increase temporal resolution (more frequent scores)
# - Avoid excessive padding

export SERVE_NUM_SEGMENTS=16    # Reduced from 32
export SERVE_SEG_LEN=8          # Reduced from 16  
export SERVE_STRIDE=4           # Reduced from 16
export SERVE_INFER_INTERVAL=2   # Inference every 2 frames (more frequent)

WINDOW_SIZE=$((SERVE_NUM_SEGMENTS * SERVE_SEG_LEN))
echo "📊 Configuration:"
echo "   Segments: $SERVE_NUM_SEGMENTS"
echo "   Frames per segment: $SERVE_SEG_LEN"
echo "   Stride: $SERVE_STRIDE"
echo "   Total window: $WINDOW_SIZE frames (~$((WINDOW_SIZE / 24))s at 24fps)"
echo "   Inference interval: Every $SERVE_INFER_INTERVAL frames"
echo ""

# Calculate coverage
echo "📏 Coverage:"
echo "   10s video (240 frames): $(((240 * 100) / WINDOW_SIZE))% window utilization"
echo "   15s video (360 frames): $(((360 * 100) / WINDOW_SIZE))% window utilization"  
echo "   20s video (480 frames): $(((480 * 100) / WINDOW_SIZE))% window utilization"
echo ""

echo "⚡ Benefits:"
echo "   ✅ Less padding for short videos"
echo "   ✅ 2x more frequent anomaly scores"
echo "   ✅ Better temporal resolution"
echo "   ✅ Faster inference (smaller window)"
echo ""

echo "⚠️  Note: This optimizes for SHORT videos (< 30s)"
echo "   For LONG surveillance videos, use default settings"
echo ""

echo "🚀 Starting dashboard with optimized settings..."
echo ""

cd "$(dirname "$0")"
./start_dashboard.sh
