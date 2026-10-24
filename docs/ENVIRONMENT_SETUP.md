# Environment Setup Documentation

## System Information
- **OS**: Windows
- **Python**: 3.12.10
- **NVIDIA GPU**: GeForce RTX 3050 6GB Laptop GPU
- **NVIDIA Driver**: 610.60
- **CUDA Runtime (PyTorch)**: 12.1
- **cuDNN**: 90100 (9.1.0)
- **Available RAM**: 16 GB
- **Node.js**: v24.16.0
- **npm**: 11.13.0
- **Git**: 2.54.0

## Project Directory
- **Location**: `C:\Users\Ganesh\sign-language-cnn`
- **Virtual Environment**: `.venv` (Python 3.12.10)

## PyTorch Installation
- **PyTorch**: 2.5.1+cu121
- **torchvision**: 0.20.1+cu121
- **torchaudio**: 2.5.1+cu121
- **CUDA Version**: 12.1 (bundled with PyTorch)
- **Installation Source**: https://download.pytorch.org/whl/cu121

## GPU Verification Results
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 90100
GPU count: 1
GPU 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Test tensor on GPU: tensor([2., 4., 6.], device='cuda:0')
Device of tensor: cuda:0
GPU test: PASSED
```

## Installed Backend Dependencies
| Package | Version |
|---------|---------|
| fastapi | 0.146.0 |
| uvicorn | 0.54.0 |
| opencv-python | 5.0.0.93 |
| opencv-contrib-python | 5.0.0.93 |
| mediapipe | 1.1.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pillow | 12.3.0 |
| python-multipart | 0.0.32 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torchaudio | 2.5.1+cu121 |

## Project Structure (Minimum)
```
sign-language-cnn/
├── .venv/
├── backend/
│   └── requirements.txt
├── docs/
│   └── ENVIRONMENT_SETUP.md
└── .gitignore
```

## Notes
- PyTorch installed with CUDA 12.1 support via official distribution
- No separate CUDA Toolkit installation required
- GPU tensor operations verified working
- Virtual environment isolated from system Python
- All dependencies installed in `.venv`