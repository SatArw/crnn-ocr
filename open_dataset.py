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
def count_keys_in_lmdb(env_path):
    env = lmdb.open(env_path, readonly=True)
    count = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            count += 1

    env.close()
    return count




with env.begin(write=True) as txn:
    # leng = txn.len()
    # bif = txn.get('image-000011506'.encode())
    # n_img = np.frombuffer(bif, dtype=np.uint8)
    # # n_img = np.reshape(n_img,(64,50))
    # n_img = n_img.reshape((160, 160))
    # print(n_img.shape)
    # img = Image.fromarray(n_img)
    # img.show()
    total_keys = count_keys_in_lmdb(root)
    print(total_keys)
    total = "11505"
    txn.put('num-samples'.encode(), total.encode())