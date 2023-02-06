import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
import faiss
from tqdm.auto import tqdm

from src.utils.nnk_solver import non_negative_qpsolver

def SingleNNKGraph(xb, xq):
    n, dim = xb.shape
    index = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(xb)

    start_time = time.time()
    similarities, indices = index.search(xq, n)
    similarities = similarities[0]
    indices = indices[0]
    
    x_support = xb[indices]
    g_i = similarities
    G_i = x_support @ x_support.T
    
    x_opt = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-6)
    weight_values = x_opt/np.sum(x_opt)

    nnk_time = time.time() - start_time
    print(f"Neighborhood sparsity: {np.count_nonzero(weight_values>1e-6)}, Time taken: {nnk_time}")

    
    return weight_values, indices