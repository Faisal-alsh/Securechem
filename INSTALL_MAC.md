# Installation Guide for macOS (Apple Silicon)

If you're on a Mac with Apple Silicon (M1/M2/M3), follow these steps:

## 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install PyTorch for Apple Silicon

```bash
# Install PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio
```

## 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `bitsandbytes` is not included as it's incompatible with macOS. The system will automatically run without 8-bit quantization on Mac.

## 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 5. Run the Server

```bash
python run_server.py
```

## Performance Notes

- **Apple Silicon Macs**: The system will use CPU/MPS backend automatically
- **Memory**: Mistral-7B requires ~14GB RAM in FP16 mode
- **Speed**: First request will be slow (model download), subsequent requests faster
- **8-bit Quantization**: Not available on Mac (requires CUDA/bitsandbytes)

## Troubleshooting

### "No module named 'torch'"

Make sure you installed PyTorch first:
```bash
pip install torch
```

### Out of Memory

The base model is large. If you run out of memory:
1. Close other applications
2. Ensure you have 16GB+ RAM
3. Consider using a smaller model or cloud GPU

### Slow Performance

First run downloads the model (~14GB), which takes time. Subsequent runs are faster.
