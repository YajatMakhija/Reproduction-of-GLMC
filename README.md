# Dual Learned Matching for Transformers

This repository contains the reproduction code for our paper on **Dual Parameterized Learned Matching** for Linear Mode Connectivity (LMC).

## Repository Structure

The code is divided into two primary domains to demonstrate the universality of our approach across both Vision and Language tasks.

### 1. Vision Transformers (`/vit`)
Contains experiments proving that Dual Learned Matching collapses catastrophic loss barriers in ViTs.
- `lmc_vit/`: The core package containing our `dual_merger.py` framework.
- `reproduce_cifar10_adv.ipynb`: Demonstrates the synergy between Clean and Adversarially Robust models.
- `reproduce_cross_domain.py`: Demonstrates Task Arithmetic by merging a sterile digit recognizer (MNIST) with a noisy real-world digit recognizer (SVHN).

### 2. Language Models (`/gpt2`)
Contains experiments adapting the framework for generative language modeling.
- `lmc_gpt2/`: The core GPT-2 LMC codebase.
- `reproduce_task_arithmetic.ipynb`: Evaluates the merging of multiple domains/skills within the GPT-2 architecture.

## Getting Started
1. Navigate to the desired directory (`cd vit` or `cd gpt2`).
2. Run the provided Python scripts or open the Jupyter/Marimo notebooks.
3. The scripts are fully self-contained and will automatically download the necessary dependencies and datasets.
