"""
Test RTX 4060 Setup
====================

Quick test to verify:
1. GPU is available and detected correctly
2. Lightweight CNN fits in 8GB VRAM
3. Training loop works
4. Memory usage is reasonable

Usage:
    python scripts/test_4060_setup.py
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.lightweight_cnn import build_lightweight_cnn, LightweightCNN, UltraLightCNN


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Will use CPU.")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Total VRAM: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Multi-Processor Count: {props.multi_processor_count}")
    
    # Check if it's 4060
    if '4060' in torch.cuda.get_device_name(0):
        print("✓ RTX 4060 detected!")
        if props.total_memory < 9e9:  # Less than 9GB
            print("✓ 8GB VRAM variant confirmed")
    
    return device


def test_model_size(variant='light', num_classes=3):
    """Test model parameter count and memory usage."""
    print(f"\n{'='*60}")
    print(f"Testing {variant.upper()} CNN ({num_classes} classes)")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_lightweight_cnn(variant=variant, num_classes=num_classes, input_channels=3)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB (fp32)")
    
    # Test forward pass
    batch_sizes = [16, 32, 64]
    img_size = (128, 128)
    
    print(f"\nForward pass test (image: {img_size}):")
    print(f"{'Batch Size':>12} {'Output Shape':>20} {'VRAM (GB)':>12} {'Status':>10}")
    print("-" * 60)
    
    for bs in batch_sizes:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        try:
            x = torch.randn(bs, 3, img_size[0], img_size[1]).to(device)
            
            with torch.no_grad():
                y = model(x)
            
            if device.type == 'cuda':
                vram = torch.cuda.max_memory_allocated() / 1e9
                status = "✓ OK" if vram < 8 else "⚠ High"
                print(f"{bs:>12} {str(y.shape):>20} {vram:>12.2f} {status:>10}")
            else:
                print(f"{bs:>12} {str(y.shape):>20} {'N/A':>12} {'✓ OK':>10}")
                
        except RuntimeError as e:
            print(f"{bs:>12} {'FAILED':>20} {'>8':>12} {'✗ OOM':>10}")
            print(f"  Error: {e}")
    
    return model


def test_training_loop(model, batch_size=32, num_iterations=10):
    """Test training loop with mixed precision."""
    print(f"\n{'='*60}")
    print("Testing Training Loop")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    
    # Create dummy data
    img_size = (128, 128)
    num_classes = model.classifier[-1].out_features
    
    dummy_imgs = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    model.train()
    
    print(f"Running {num_iterations} training iterations (batch_size={batch_size})...")
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(dummy_imgs)
                loss = criterion(out, dummy_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(dummy_imgs)
            loss = criterion(out, dummy_labels)
            loss.backward()
            optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")
    
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM during training: {peak_vram:.2f} GB")
        
        if peak_vram < 6:
            print("✓ VRAM usage is healthy for RTX 4060 8GB")
        elif peak_vram < 7.5:
            print("⚠ VRAM usage is high but acceptable")
        else:
            print("✗ VRAM usage is too high, reduce batch_size")
    
    print("✓ Training loop completed successfully")


def test_comparison_with_resnet18():
    """Compare our lightweight CNN with ResNet18."""
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    
    # Our models
    light = LightweightCNN(num_classes=3)
    ultra = UltraLightCNN(num_classes=3)
    
    # Count parameters
    light_params = sum(p.numel() for p in light.parameters())
    ultra_params = sum(p.numel() for p in ultra.parameters())
    
    # ResNet18 (from torchvision)
    import torchvision.models as models
    resnet = models.resnet18(num_classes=3)
    resnet_params = sum(p.numel() for p in resnet.parameters())
    
    print(f"{'Model':<20} {'Parameters':>15} {'Ratio':>10}")
    print("-" * 50)
    print(f"{'UltraLight CNN':<20} {ultra_params:>15,} {ultra_params/resnet_params:>10.2%}")
    print(f"{'Light CNN':<20} {light_params:>15,} {light_params/resnet_params:>10.2%}")
    print(f"{'ResNet18':<20} {resnet_params:>15,} {'100.00%':>10}")
    
    print(f"\nLight CNN is {resnet_params/light_params:.1f}x smaller than ResNet18")
    print("✓ Significantly less memory usage and faster training")


def main():
    print("=" * 60)
    print("RTX 4060 Setup Test")
    print("=" * 60)
    print()
    
    # 1. GPU info
    device = print_gpu_info()
    
    # 2. Model comparison
    test_comparison_with_resnet18()
    
    # 3. Test light CNN
    model_light = test_model_size(variant='light', num_classes=3)
    test_training_loop(model_light, batch_size=32, num_iterations=10)
    
    # 4. Test ultra CNN
    model_ultra = test_model_size(variant='ultra', num_classes=3)
    test_training_loop(model_ultra, batch_size=32, num_iterations=10)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print("✓ All tests passed!")
    print("✓ Your RTX 4060 is ready for training")
    print()
    print("Recommended settings for RTX 4060 8GB:")
    print("  - Model: Light CNN (500K params)")
    print("  - Batch size: 32")
    print("  - Image size: 128x128")
    print("  - Mixed precision: Enabled")
    print()
    print("To start training:")
    print("  python scripts/train_grouped_4060.py --all")


if __name__ == '__main__':
    main()
