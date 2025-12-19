Content
* Problem Setting
* Formal


# Problem Setting

reconstruct missing regions in images with
Learning category
Input
Output
Learning objective


Limitations


## Problem
## Learning Category
## Input
## Output
## Learning Objective
## Evaluation Metrics
* IoU (Intersection over Union)
* Dice Coefficient.
**goal:** maximize the area of overlap between the predicted pixel and the ground truth pixel divided by their union. 
# Explore Dataset
# Masking Strategy
## Random strokes
## Rectangles (variable size?)
# Basic Pipeline
## Setup & Producibility
## Data Load and Split
## Data Preprocessing (Train) + Masking
        Images on disk
                ↓
        BaseDataset (ToTensor only)
                ↓
        Split into train / val
                ↓
        Compute mean & std on train
                ↓
        Define Normalize(mean, std)
                ↓
        Final datasets (ToTensor + Normalize)
                ↓
        Wrap for task (inpainting)
                ↓
        DataLoaders
### resize/crop/normalize (mean/std or [-1, 1]) / mask
## Baseline
a) Navier-Stokes based Inpainting (2001): “Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting” (https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf)
b) Fast Marching Method based: https://www.olivier-augereau.com/docs/2004JGraphToolsTelea.pdf
Often, such traditional methods rely on the solving of
Partial Differential Equations (PDEs). Examples of such approaches are the inpainting
method by Telea and inpainting based on the Navier-Stokes PDEs

Implemented averaging neighbour: 
This algorithm iteratively fills missing pixels by averaging their spatial neighbors while keeping known pixels fixed, which corresponds to solving a discrete diffusion equation with Dirichlet boundary conditions.”
## Model Architecture
### U-Net
## Training
## Model Evaluation
Evaluation protocol (what you should do)
* Fix a single deterministic mask per image (your idx+seed approach).
* For each method (diffusion baseline, CNN/U-Net): produce pred from masked input.
* Compute metrics:
   * MAE_miss
   * PSNR_miss
* Report: mean ± std over the test set; plus a few qualitative examples (same indices for all methods)
### Success Metrics
## hyperparameter search


**Resources**:
* [Basic but clean data pipeline / intro:](https://apxml.com/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/customizing-dataloader-behavior)