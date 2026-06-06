# Synthetic OCR Image Generation Engine

An automated, high-performance Python pipeline designed to generate, augment, and split synthetic text-line images for training Optical Character Recognition (OCR) systems from scratch. 

The engine handles everything from character map validation to complex geometric warping, compiling **43,500 highly variable tokens in under 2 minutes** on modern hardware (like an M-series MacBook Air).

---

## Features

### 1. Font Validation & Safety Filtering
* Utilizes `fontTools` to analyze system `.ttf` / `.otf` files before processing.
* Dynamically filters out corrupted fonts or families lacking core Latin-1/ASCII character support to prevent empty glyph rendering.

### 2. Multi-Mode Text Engine with Weighted Probability
* Rejects static text strings in favor of deep structural variety.
* Uses controlled weighted randomness (`random.choices` / `np.random.choice`) to distribute text configurations across three distinct profiles:
  * **Dictionary Words:** Common vocabulary text strings.
  * **Full Sentences:** Chains of words utilizing unpredictable structural capitalization and trailing punctuation markers.
  * **Gibberish Strings:** Random, non-linguistic character sequences that force the model to prioritize raw pixel feature extraction over language sequence prediction.

### 3. Perception-Based Contrast Control
* Dynamically calculates foreground (text) and background color pairings.
* Evaluates pairs using human-perceived luminance weights:  
  `Brightness = 0.299*R + 0.587*G + 0.114*B`
* Automatically discards and rerolls combinations failing to hit a baseline delta threshold of `>100`, guaranteeing readable training contrast while maximizing color diversity.

### 4. Heavy Augmentation Pipeline
Applies a randomized degradation chain to simulate real-world document artifacts:
* **Wave Distortion:** Sinusoidal grid transformations to warp text lines along arbitrary paths.
* **Ink Dynamics:** Morphological erosion/dilation layers to emulate ink bleeding or print fading.
* **Noise Injection:** Layered salt & pepper artifacts and custom-scaled Gaussian blurs.

### 5. Font-Isolated Train/Test Splitting
* **Prevents Data Leakage:** Instead of a generic random image shuffle, the engine partitions data by the **fonts themselves**.
* Selects 80% of unique system fonts exclusively for `train.csv` and seals the remaining 20% into `test.csv`.
* Ensures the evaluation set tests true zero-shot font generalization.

---

## Project Structure

```text
synthetic_image generation/
├── output/
│   ├── train/               # ~34,711 augmented PNG images
│   ├── test/                # ~8,789 augmented PNG images
│   ├── train.csv            # Training log metadata (path, text, font_name)
│   └── test.csv             # Evaluation log metadata (path, text, font_name)
├── main.py                  # Core execution engine
├── README.md                # Codebase documentation
└── pyproject.toml         # Dependency Manifest