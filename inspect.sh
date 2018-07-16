#!/bin/sh

python3 testgen.py

python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/inspect_checkpoint.py --file_name=./output/vgg16.ckpt
