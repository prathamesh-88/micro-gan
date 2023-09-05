import os
import numpy as np

file_dir = os.getcwd()
seed_file_path = os.path.join(file_dir, "utilities", "seed.npy")
seed = np.load(seed_file_path)

# seed = tf.random.normal([32, 250])
# seed_numpy = seed.numpy()
# np.save(seed_file_path, seed_numpy)
