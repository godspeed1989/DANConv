import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors
import time

batch_size = 16
num_points = 81920
K = 30
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours
start = time.time()
neigh_idx = nearest_neighbors.knn_batch(pc, pc, K, omp=True)
print(time.time() - start)
print(neigh_idx.shape) # (B, N, K)

start = time.time()
gather_l = []
for b in range(batch_size):
    gather_l.append(pc[b][neigh_idx[b].reshape(-1)])
gathered = np.stack(gather_l, axis=0)
gathered = gathered.reshape(batch_size, num_points, K, 3)
centeroid = pc.reshape(batch_size, num_points, 1, 3)
gathered = gathered - centeroid
print(gathered.shape) # (B, N, K, 3)
print(time.time() - start)


