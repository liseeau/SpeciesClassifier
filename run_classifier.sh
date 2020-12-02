#!/bin/bash
# Author: Shiya Li (sl6719@imperial.ac.uk)
# Desc: run classifier.py
# Arguments: 
# --species: species to be classified (required); --pred_dir: relative path of images for prediction (required); 
#--train_dir: relative path of images for training; --model_dir: relative path of pretrain model.
# Date: Oct 2020

# Sample input: 

#python classifier.py --species human --pred_dir ../test_images/ --train_dir ../train_images --model_dir imagenet

python classifier.py --species human --pred_dir ../pred --train_dir ../train --augmentation True --aug_size 1 --sample_size 50 --model MLP

if [ -d "../train_processed" ]; then
rm -r ../train_processed 
fi

if [ -d "../pred_processed" ]; then
rm -r ../pred_processed
fi