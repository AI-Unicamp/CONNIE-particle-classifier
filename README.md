# CONNIE Particle Classifier

**Machine learning and image processing tools for particle classification in the CONNIE experiment using Skipper-CCD sensor data.**

## Overview

This repository contains tools and models developed to classify particle tracks captured in the CONNIE (Coherent Neutrino-Nucleus Interaction Experiment) detector. The classification focuses on identifying events such as muons, electrons, blobs, diffusion hits, alphas, and others, using both image-based and feature-based machine learning techniques.

The main objective is to support the detection and study of **coherent elastic neutrino-nucleus scattering (CEνNS)** by improving the signal-to-background separation in CONNIE’s Skipper-CCD images.

---

## Features

- 📦 **Event cropping** and preprocessing from Skipper-CCD energy-calibrated FITS images  
- 🧠 **Convolutional Neural Network (CNN)** classifier (ResNet-18) trained on labeled and augmented event images  
- 🌲 **Feature-based models** (Random Forest, XGBoost) using event metadata extracted from ROOT catalogs  
- 🧪 Cross-validation, grid search, and accuracy benchmarking  
- 🖼️ GUI for human labeling with Annotation Redundancy & Quality Assurance  
- 📊 Evaluation tools and confusion matrix reporting

---

## Project Structure

```
├── data/                # Event images and raw data
├── database             # Database related files
├── models/              # Trained model checkpoints
├── scripts/             # Image processing and training scripts
├── gui/                 # Annotation GUI tools
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── results/             # Evaluation results and figures
├── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- OpenCV
- NumPy, Matplotlib, Scikit-learn
- XGBoost, ROOT, uproot
- Tkinter (for GUI annotation tool)

### Installation

```bash
git clone https://github.com/yourusername/connie-particle-classifier.git
cd connie-particle-classifier
pip install -r requirements.txt 
```

You can also create a virtual environment
```bash
git clone https://github.com/yourusername/connie-particle-classifier.git
cd connie-particle-classifier
create_venv.sh
source virtualenv/bin/activate
```

---

## Usage

### 1. Preprocess and Crop Events

```bash
python scripts/extract_events.py --input_folder raw_images/ --output_folder data/events/
```

### 2. Train CNN Model

```bash
python scripts/train_cnn_credo.py
```

### 3. Launch GUI Labeling Tool

```bash
python gui/data_label.py
```

---

## Dataset

- **Raw Data**: Skipper-CCD images in FITS format from CONNIE Run 125  
- **Labeled Subset**: Annotated events (PNG and ROOT) from Runs 118 and 125  
- **External**: [CREDO dataset](https://credo.science) used for transfer learning experimentation

> **Note**: Due to collaboration restrictions, some datasets may not be publicly available.

---

## Results

- CNN classification accuracy on test set (CREDO dataset): **~95%**
- Feature-based classifiers (XGBoost, RF) with CONNIE dataset: **~88–89%**
- Manual annotation strategy shows improved reliability using redundancy

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Developed as part of research with the CONNIE Collaboration.
