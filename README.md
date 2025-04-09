# Adversarial Attack Framework

This project implements various adversarial attack methods for object detection models, including FGSM, PGD, CW, BIM, DeepFool, and ZOO. The framework is designed to be modular and extensible, allowing easy integration of new attack methods.

## Project Structure

- `configs/`: Configuration files for different attack methods.
- `core/attacks/`: Implementation of adversarial attack methods.
- `core/models/`: Wrapper for the YOLO model with adversarial attack support.
- `core/utils/`: Utility functions for image processing, visualization, and configuration management.
- `main.py`: Entry point for running attacks and visualizing results.

## Supported Attack Methods

- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **CW**: Carlini & Wagner Attack
- **BIM**: Basic Iterative Method
- **DeepFool**: Minimal perturbation attack
- **ZOO**: Zeroth Order Optimization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/adversarial-attack-framework.git
   cd adversarial-attack-framework