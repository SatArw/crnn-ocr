import lmdb
from PIL import Image
import numpy as np

root = "./data/train"
env = lmdb.open(root,
            max_readers=1,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False)

with env.begin(write=True) as txn:
    # leng = txn.len()
    bif = txn.get('image-000000879'.encode())
    n_img = np.frombuffer(bif, dtype=np.uint8)
    # n_img = np.reshape(n_img,(64,50))
    n_img = n_img.reshape((160, 160))
    print(n_img.shape)
    img = Image.fromarray(n_img)
    img.show()