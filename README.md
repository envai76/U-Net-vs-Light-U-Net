In this repository, we compare the performance of two networks( U-Net and Light-U-Net) on our subjective problem.

# Project Title
A Hybrid Framework for Effective Microscopic Cell Counting & Segmentation Integrating Light-U-net with Watershed

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)
- [Contact](#contact)

---


## Overview

This project implements blob detection algorithms (DoG, LoG, DoH) for cell counting in microscopy images.

---

## Features
- Detects cells using multiple algorithms.
- Adjustable parameters.
- Generates outputs for validation.

---

## Installation
conda create -n tf-gpu-env tensorflow-gpu numpy=1.23.5


To run the training phase of this code, you'll need to set up a specific environment with the required dependencies. Follow these steps to create the environment:

1. Create the conda environment with the required TensorFlow version and additional dependencies:

   ```bash
   conda create -n tf-gpu-env tensorflow-gpu numpy=1.23.5

2. Activate the environment:

   ```bash
    conda activate tf-gpu-env
Once the environment is set up and activated, you can proceed with running the training code.

# Refrences
1. N. Y. Gharaei, N. Gaikwad, D. Upadhyay, S. Sampalli, B. C. Chauhan, and A. J. Jamet. Comparative evaluation of deep learning architectures for retinal ganglion cell counting: FCRN-A, FCRN-A-v2, and U-Net. In 2024 International Conference on Machine Learning and Applications (ICMLA), Miami, FL, USA, Dec. 2024. Accepted for publication.
