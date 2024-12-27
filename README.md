
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
2. **TensorFlow 2.x** (using `v1` compat mode). For example:
   - `pip install "tensorflow>=2.8,<3"`
3. **NumPy**, **Matplotlib**, **opencv-python**, **scipy**, **scikit-image** (for `ssim`), and **pickle** (built in to Python).
4. A **GPU** is optional but recommended (code was tested on an RTX 30-series GPU). CPU-only usage is possible but significantly slower.

Example environment setup on Windows (using `pip`):
```bash
pip install numpy matplotlib opencv-python scikit-image scipy tensorflow
```

---

## 3. Directory Structure

Ensure your folder layout matches the references in `main.py`. By default, the script expects:

```
C:\Users\mreza\OneDrive\Documents\GitHub\Attention-VGG\Data
 ├── catbins\
 ├── object_GradsTCs\
 ├── images\
 ├── ori_catbins\  (optional)
 ├── vgg16_weights.npz
 ├── ...
 ├── main.py
 ├── vgg_16.py
 ├── attention.py
 ├── utils.py
 ...
```

Inside **`object_GradsTCs`** are your `.txt` files for gradient/tuning data:
- `CATgradsDetectTrainTCs_im1.txt`, `CATgradsDetectTrainTCs_im2.txt`, etc.
- `featvecs20_train35_c.txt` for tuning-based attention.

Inside **`images`** are your `.npz` images, for example:
- `arr5_c0.npz`, `arr5_c5.npz`, `merg5_c5.npz` ...
- possibly also `cats20_test15_c.npy` for test images.

Inside **`catbins`** are your TensorFlow checkpoint files for category-specific binary classifiers, e.g.:
- `catbin_5.ckpt.data-00000-of-00001`, `catbin_5.ckpt.index`, etc.

---

## 4. Running the Code

1. **Clone** this repository or copy the four main Python files (`main.py`, `vgg_16.py`, `attention.py`, `utils.py`) into your working directory.

2. **Set environment variables** or edit `setup_paths()` in `main.py` if your data root folder differs. For instance:
   ```python
   def setup_paths():
       base_path = r'C:\Users\mreza\OneDrive\Documents\GitHub\Attention-VGG\Data'
       return {
           'tc_path': os.path.join(base_path, 'object_GradsTCs'),
           'weight_path': base_path,
           'image_path': os.path.join(base_path, 'images'),
           'save_path': base_path
       }
   ```

3. **Run** in a terminal (cmd or PowerShell) from the same folder:
   ```bash
   python main.py --imtype 1 --category 5 --layer 10 --attention_type TCs --batch_size 1 --max_images 2
   ```
   - **`--imtype`**: image type (1 = merged, 2 = array, 3 = test).
   - **`--category`**: integer in range [0..19], for the object category to analyze.
   - **`--layer`**: which VGG16 conv layer (0..12) or >12 for "all layers" scaled 0.1.
   - **`--attention_type`**: either `TCs` (tuning-based) or `GRADs` (gradient-based).
   - **`--batch_size`**: how many images to process at once.
   - **`--max_images`**: how many images to load from the .npz files.

4. The script will:
   1. Load the pretrained `vgg16_weights.npz`.
   2. (Optionally) load a category-specific checkpoint from `catbins/catbin_{category}.ckpt`.
   3. Load images from e.g. `images/arr5_c5.npz` or `images/merg5_c5.npz`.
   4. Generate attention maps for each conv layer based on your chosen approach.
   5. Compute saliency maps, compare them to attention maps, and store metrics (Pearson, IoU, SSIM, KL).
   6. Save results to `attention_results_cat5_layer10/...` or a similar subfolder.
   7. Optionally produce **PNG** visualizations showing Original / Saliency / Attention side by side plus a histogram of saliency intensities.

---

## 5. Explanation of Key Scripts

### 5.1 `attention.py`
- `AttentionMechanism`:  
  Creates per-layer attention masks in either **multiplicative** or **additive** mode. The `make_tuning_attention` function loads `featvecs20_train35_c.txt`, while `make_gradient_attention` function loads `CATgradsDetectTrainTCs_imN.txt`.

- `LayerAttention`:  
  A helper class to generate a mask of strengths across the 13 VGG16 conv layers, or scale attention for a single layer, etc.

### 5.2 `main.py`
- **Command-line arguments** parse your chosen category, image type, layer, etc.
- **`setup_paths()`** points to your local data directories.
- **Load** VGG16 base, try to restore a `catbin_{X}.ckpt`, then load images from `.npz`.
- Runs the pipeline:
  1. **Generate attention maps** with `make_attention_maps_with_batch`.
  2. **Compute saliency** with `compute_saliency_map`.
  3. **Compare** with `compare_saliency_attention`.
  4. **Visualize** using `visualize_comparison`.
  5. **Save** results as `.npy`.

### 5.3 `utils.py`
- `AttentionAnalyzer`: Optionally computes the effect of attention on true pos / true neg sets, if you do a binary classification approach.
- `pad_batch`: zero-pads images if the final batch is smaller than `batch_size`.
- `compare_saliency_attention`: alignment metrics between saliency and attention.
- `visualize_comparison`: draws side-by-side subplots with Original, Saliency, and averaged Attention.

### 5.4 `vgg_16.py`
- A custom VGG16 architecture that introduces placeholders **`a11, a12, a21, ...`** for attention at each conv layer. Then the conv output is multiplied by that attention mask (`tf.multiply(...)`).

---

## 6. Frequently Asked Questions (FAQ)

1. **Why do I get a mismatch with shapes like `(batch,224,224,3)`?**  
   - Confirm your `.npz` files produce the correct shape. Some `.npz` might have `(15, 224,224,3)` which we reshape or slice for `batch_size`.

2. **Pickle/Encoding errors**  
   - If you see `UnicodeDecodeError` or similar when loading `.txt` with pickle, try opening in `latin1` mode, e.g.:
     ```python
     with open(grad_file, "rb") as fp:
         grads = pickle.load(fp, encoding='latin1')
     ```
   - The sample code attempts normal `pickle.load(fp)` so you may need to modify if your environment differs.

3. **What if the catbin checkpoint doesn’t exist?**  
   - The code prints a warning, skipping the checkpoint. It can still run using just the universal `vgg16_weights.npz`.

4. **Which TensorFlow version is recommended?**  
   - Any TF 2.x (like `2.8 <= TF < 3.0`). You can also do `import tensorflow.compat.v1 as tfv1` and `tfv1.disable_eager_execution()` if needed. The code uses placeholders, so it’s mostly TF 1.x style in v2 compat mode.

5. **What about multi-GPU or HPC usage?**  
   - This code is baseline single-GPU. Extending for HPC data parallel training is beyond scope but feasible in TF2 or PyTorch.

---

## 7. License and Citation

- These scripts incorporate code from the original VGG16 tutorials and from standard interpretability references.  
- If you build upon or publish results, please give credit to the **original VGG authors** and any relevant interpretability references.

---

## 8. Contact

- For any question regarding the code or the dataset structures, feel free to create an Issue or contact:
  - **Haochen Yang** at [hy2611@nyu.edu](mailto:hy2611@nyu.edu)
  - **Michael R. Amiri** at [ma7422@nyu.edu](mailto:ma7422@nyu.edu)

We hope you find this project helpful for exploring attention-based interpretability in VGG16 networks!
```
