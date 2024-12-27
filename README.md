# Attention-VGG Project

This repository implements a modified VGG16 model that can apply **attention masks** at each convolutional layer. The attention maps are derived either from **tuning curves** or **gradient-based** signals, and are compared to **saliency maps** to investigate interpretability.

Below you will find detailed instructions on how to set up the project, run the code, and how each component of the repository is organized.

---

## 1. Overview

- **`attention.py`**: Defines the `AttentionMechanism` class for creating attention masks based on either **tuning curves** or **gradient** files, as well as the `LayerAttention` helper for per-layer attention control.
- **`main.py`**: The primary entry-point script that parses command-line arguments, initializes the VGG16 model, loads data from your local directories, and runs the analysis pipeline (saliency computation, attention map generation, comparisons, and optional visualizations).
- **`utils.py`**: Contains helper functions and classes:
  - `AttentionAnalyzer` to compute responses with or without attention,
  - `pad_batch` utility for zero-padding smaller batches,
  - `compute_saliency_map` for gradient-based saliency,
  - `compare_saliency_attention` to compare saliency and attention maps with correlation/IoU/SSIM/KL, 
  - `visualize_comparison` to generate side-by-side images.
- **`vgg_16.py`**: A specialized version of the VGG16 network with placeholder-based attention multiplications. It sets up each conv layer as `convX_Y` with placeholders `aXy` for attention.

---

## 2. Requirements

1. **Python 3.8+** (recommended)
2. **TensorFlow 2.x** (using v1 compat mode). For example:
   - `pip install "tensorflow>=2.8,<3"`
3. **NumPy**, **Matplotlib**, **opencv-python**, **scipy**, **scikit-image** (for `ssim`), and **pickle** (built in to Python).
4. A **GPU** is optional but recommended (code was tested on an RTX 30-series GPU). CPU-only usage is possible but significantly slower.

Example environment setup on Windows (using `pip`):
```bash
pip install numpy matplotlib opencv-python scikit-image scipy tensorflow
