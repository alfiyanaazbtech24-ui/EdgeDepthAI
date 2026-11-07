
# EdgeDepth AI

## Edge-Preserving Depth Map Upsampling and Refinement

EdgeDepth AI is a computer vision and deep learning
application for reconstructing high-resolution depth
maps from sparse, noisy and low-resolution depth data.

## Main Features

- Classical depth interpolation
- Joint bilateral filtering
- Guided filtering
- Edge detection
- Depth discontinuity detection
- Hole filling
- Boundary-aware refinement
- RGB-guided PyTorch depth reconstruction
- Confidence-aware hybrid reconstruction
- 2x, 4x and 8x upsampling
- NYU Depth V2, KITTI and Middlebury support
- Quantitative evaluation metrics

## Technologies

- Python
- Flask
- OpenCV
- NumPy
- SciPy
- PyTorch
- scikit-image
- Bootstrap
- JavaScript
- Chart.js

## Installation

Create a virtual environment:

    py -3.11 -m venv .venv

Activate it:

    .venv\Scripts\activate

Upgrade pip:

    python -m pip install --upgrade pip

Install dependencies:

    pip install -r requirements.txt

## Initialize Configuration

    python config.py

## Run Application

    python app.py

Open:

    http://127.0.0.1:5000

## Project Status

Phase 1: Project foundation
Phase 2: Classical interpolation and preprocessing
Phase 3: Edge-aware classical algorithms
Phase 4: Deep learning architecture
Phase 5: Dataset integration
Phase 6: Model training and evaluation
Phase 7: Hybrid reconstruction
Phase 8: Flask REST API
Phase 9: Frontend integration
Phase 10: Testing and deployment

A trained model checkpoint is required for
deep learning inference. Classical methods
can operate independently.
