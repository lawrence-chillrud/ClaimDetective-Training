# ClaimDetective-Training

This repo contains the code used to train [ClaimDetective](https://github.com/lawrence-chillrud/ClaimDetective), a check-worthiness / claim detection classification model.

## Overview

1. [roberta.py](roberta.py) contains the code to train and test the claim detection model.

2. [source](source) is a directory containing other source code used to help `roberta.py` run. 

    * [ernie.py](source/ernie.py) contains basic helper functions for argument parsing and output formatting etc.
    * [models.py](source/models.py) contains the model architecture. 
