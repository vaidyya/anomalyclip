#!/usr/bin/env python
"""
Test script to verify UCF-Crime model configuration
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ["PROJECT_ROOT"] = str(project_root)

print("🔍 Testing UCF-Crime Model Configuration\n")
print("=" * 60)

# Test config discovery
from src.server.config import discover_runtime, load_class_names

try:
    checkpoint_path, labels_file, arch, normal_id = discover_runtime()
    
    print("\n✅ Configuration Discovery Successful!\n")
    print(f"📦 Checkpoint: {Path(checkpoint_path).name}")
    print(f"   Full path: {checkpoint_path}")
    print(f"\n🏷️  Labels: {Path(labels_file).name}")
    print(f"   Full path: {labels_file}")
    print(f"\n🏗️  Architecture: {arch}")
    print(f"🎯 Normal class ID: {normal_id}")
    
    # Load and display class names
    class_names = load_class_names(labels_file)
    print(f"\n📋 Classes ({len(class_names)} total):")
    for i, name in enumerate(class_names):
        marker = " ← NORMAL" if i == normal_id else ""
        print(f"   {i}: {name}{marker}")
    
    # Verify it's UCF-Crime
    print("\n" + "=" * 60)
    if "ucf" in checkpoint_path.lower():
        print("✅ UCF-Crime model is configured correctly!")
        print("\nYou can now run the dashboard with:")
        print("  ./start_dashboard.sh")
    else:
        print("⚠️  Warning: Not using UCF-Crime checkpoint")
        print(f"   Current: {checkpoint_path}")
        print("\nTo force UCF-Crime, set environment variable:")
        print("  export ANOMALY_CLIP_CKPT=checkpoints/ucfcrime/last.ckpt")
    
except Exception as e:
    print(f"\n❌ Configuration Error: {e}")
    print("\nPlease ensure:")
    print("  1. Checkpoint exists: checkpoints/ucfcrime/last.ckpt")
    print("  2. Labels exist: data/ucf_labels.csv")
    sys.exit(1)

print("\n" + "=" * 60)

# Test PyTorch and MPS
print("\n🔍 Checking PyTorch and Device Support...\n")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available - GPU acceleration enabled")
        device = "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA available")
        device = "cuda"
    else:
        print("ℹ️  Using CPU (MPS/CUDA not available)")
        device = "cpu"
    
    print(f"\n🖥️  Will use device: {device}")
    
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch torchvision")
    sys.exit(1)

print("\n" + "=" * 60)
print("\n✨ All checks passed! Ready to run dashboard.\n")
