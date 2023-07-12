import os
import lmdb
import six
import sys
import numpy as np
from PIL import Image
import torch

def create_lmdb_dataset(image_dir, label_dir, output_path):
    # Create LMDB environment
    env = lmdb.open(output_path, map_size=int(1e11))

    # Start write transaction
    with env.begin(write=True) as txn:
        # Iterate over images in the image directory
        for i, filename in enumerate(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename)
            label_path = label_path.replace('.jpg','.txt')

            # Read and preprocess image
            image = Image.open(image_path).convert('L')
            image_bytes = image.tobytes()

            # Generate unique key for the image
            key = 'image-%09d' % (i + 1)

            # Store image in LMDB database
            txn.put(key.encode(), image_bytes)

            # Read label from label file
            with open(label_path, 'r') as label_file:
                label = label_file.read().strip()

            # Store label in LMDB database
            label_key = 'label-%09d' % (i + 1)
            txn.put(label_key.encode(), label.encode())
        total = f"{i+1}"
        print(total)
        txn.put("num-samples".encode(),total.encode())
    # Close the LMDB environment
    env.close()

# Example usage:
image_dir = '/home/satarw/data_ocr/train_imgs'
label_dir = '/home/satarw/data_ocr/train_lbl'
output_path = './data/train'

create_lmdb_dataset(image_dir, label_dir, output_path)
