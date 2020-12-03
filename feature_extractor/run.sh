#!/bin/bash
python extract_pretrained_features.py --model vgg16 --patches 1 --folds 1 --height 590 --width 600 -i ../data/processed_images/f%d/*.bmp -o ../data/features/fold-%d_patches-%d.npy
