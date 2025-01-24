In this repository, we compare the performance of two networks( U-Net and Light-U-Net) on our subjective problem.

# Project Title
A Hybrid Framework for Effective Microscopic Cell Counting & Segmentation Integrating Light-U-net with Watershed

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#Dataset)
- [Pre-processing](#Pre-processing)
- [Deep Learning Models](#Deep-Learning-Models)
- [Train on your own Dataset](#Train-on-your-own-Dataset)
- [Methodology](#Methodology)
- [Counting Methodologies](#Counting-Methodologies)
- [Evaluation](#Evaluation)
- [License](#refrences)
- [Contact](#contact)

---


## Overview

This repository offers an optimized deep learning method for retinal ganglion cell (RGC) counting and segmentation. It introduces advanced pre-processing techniques for both input and label datasets, along with an efficient Light-U-Net model. The repository includes a synthetic dataset and a self-generated dataset of whole-mounted mouse retina images. 
We evaluate the proposed Light-U-Net model against the standard U-Net across both datasets and compare various counting and segmentation methods. Two types of label pre-processing are used to generate region and centroid label sets. 
Local maxima, connected component, watershed, and feature-based counting methods (LoG, DoG, DoH) are applied to both Light-U-Net and U-Net predictions to facilitate a fair comparison. 
The results demonstrate the model's robustness and suitability for RGC analysis under varying imaging conditions.

---


## Installation

To run the training phase of this code, you'll need to set up a specific environment with the required dependencies. Follow these steps to create the environment:

1. Create the conda environment with the required TensorFlow version and additional dependencies:

   ```bash
   conda create -n tf-gpu-env tensorflow-gpu numpy=1.23.5

2. Activate the environment:

   ```bash
   conda activate tf-gpu-env

3. Once the environment is set up and activated, you can proceed with installing the  dependencies  and then training the code.

   ```bash
   bash install_dependencies.sh
   ```

    Or you can use:

    ```bash
    pip install -r requirements.txt
    ```

---


## Dataset

Both real and synthetic datasets are provided in ***real_dataset*** and ***synth_dataset*** folder of  this repository. 
The custom dataset, collected by Dalhousie University’s Department of Ophthalmology.
the subfolders in each include :

1. cells : input images
2. dots : annotated label images

## Pre-processing 
All pre-processing steps are visualized in ***visulize the preprocessing steps*** folder.

![Pre_processing steps](images/Preprocessing.png)  
*Figure 1: Input and centroids/regions label image pre-processing steps*

----


## Deep Learning Models

The data augmentation code is provided in ***generator.ipynb*** file. The models architectures are provided in ***model.ipynb*** file.
![U-Net](images/U_Net.jpg)  
*Figure 2: Architecture of U-Net*

![Light-U-Net](images/Light_UNet.jpg)  
*Figure 3: Architecture of Light-U-Net*

---

## Train on your own Dataset

Both the training codes of Light-U-Net and U-net are provided in ***Light-U-Net.ipynb*** and ***U-Net.ipynb*** respectively.

---

##  Methodology

The methodology is presented below:  
![Methodology](images/framework.png)  
*Figure 4: Framework of our proposed methodology*

---

## Counting Methodologies 
All counting methodologies are provided under the ***counting codes*** folder, including:
- Local Maxima
- Connected Component
- Watershed
- LoG (Laplacian of Gaussian)
- DoG (Difference of Gaussian)
- DoH (Determinant of Hessian)

The output results are programmed to be saved in the dataset folder.


---


## Evaluation
All evaluation techniques, including ICC and IoU, are available in the ***evaluation methods*** folder. The output results are configured to be saved in the same directory. Additionally, the performance calculations for counting are included within each respective counting script in the ***counting codes*** folder.

---

## Refrences

1. N. Y. Gharaei, N. Gaikwad, D. Upadhyay, S. Sampalli, B. C. Chauhan, and A. J. Jamet. Comparative evaluation of deep learning architectures for retinal ganglion cell counting: FCRN-A, FCRN-A-v2, and U-Net. In 2024 International Conference on Machine Learning and Applications (ICMLA), Miami, FL, USA, Dec. 2024. Accepted for publication.

2. O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, Springer, 2015, pp. 234–241.

3. N. Y. Gharaei, N. Gaikwad, D. Upadhyay, B. C. Chauhan, and S. Sampalli,  ``A Hybrid Framework for Effective Microscopic Cell Counting \& Segmentation Integrating Light-U-net with Watershed: in progress,'' \textit{IEEE Transactions on Medical Imaging}, in progress, 2025.

----

## Contact
Let me know if you need help with any specific section or adding more details! you can contact with us using the following email address:
1. Narges.yarahmadi@dal.ca

