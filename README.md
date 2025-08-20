# Offline Qwen Image Edit Tool

This is a fully offline version of the Qwen/Qwen-Image-Edit model demo using Gradio.
No external API keys are required. It runs locally using PyTorch + Diffusers on your GPU.

---

## ✅ Features
- Local image editing using the Qwen Image Edit model
- Simple Gradio interface
- No Dash API key or internet calls
- Works with CUDA GPU (required for reasonable speed)

---

## ⚠ Requirements

- Python 3.9 or 3.10
- CUDA-capable GPU with ~8GB+ VRAM
- Stable internet only for first-time model download

---
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fanu2/offline-qwen-image-edit/blob/main/offline_qwen_image_edit.ipynb)

## Installation

```bash
git clone https://github.com/yourname/offline-qwen-image-edit.git
cd offline-qwen-image-edit

# create virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt
