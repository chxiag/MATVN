# MATVN

PyTorch implementation of MATVN described in the paper entitled "Multi-Scale Adaptive Attention-based Time-Variant Neural Networks for Multi-Step Time Series Forecasting" by Changxia Gao, Ning Zhang, Youru Li, Yan Lin and Huaiyu Wan.


The key benifits of MATVN are:
1. It is a novel time-variant model that learns dynamic irregular temporal behavior over time.
2. The newly proposed Multi-scale Multi-head Diverse attention (MMD) module makes the prediction model encode time series information on different temporal scales.
3. A new Adaptive Window-aware Mask strategy is introduced into MMD module to improve the ability of our proposed model to adaptively learn different attention range representations for different tokens in time series, and successfully enhance the flexibility of the model.
4. It is shown to out-perform statistical model and state of the art deep learning models.
## Files


- MATVN.py: Contains the main class for MATVN (Multi-scale Adaptive attention-based Time-Variant neural Networks).
- TVarchitecture.py: Contains the main classes for Time-variant architecture and Multi-scale Multi-head Diverse attention module.
- calculateError.py: Contains helper functions to compute error metrics
- dataHelpers.py: Functions to generate the dataset.
- demo.py: Trains and evaluates MATVN.
- evaluate.py: Contains a rudimentary training function to train MATVN.
- optimizer.py: Implements Open AI version of Adam algorithm with weight decay fix.
- train.py: Contains a rudimentary training function to train MATVN.

## Data
All the data in the paper can be found here https://arxiv.org/pdf/2012.07436.pdf.

## Usage

Run the demo.py script to train and evaluate MATVN model. 

## Requirements

- Python 3.6
- Torch version 1.2.0
- NumPy 1.14.6.

