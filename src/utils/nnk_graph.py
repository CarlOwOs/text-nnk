import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
import faiss
from tqdm.auto import tqdm

from src.utils.nnk_solver import non_negative_qpsolver

def nnk_graph(features, top_k, kernel, cuda_available):
    n, dim = features.shape
    index = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)

    start_time = time.time()
    similarities, indices = index.search(features, top_k+1)
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]

    #sigma_sq = 0.5 * np.mean(similarities[:, 1:]) ** 2
    
    weight_values = np.zeros((n, top_k))

    for i, x in enumerate(tqdm(features)):
        neighbor_indices = indices[i, :]
        x_support = features[neighbor_indices]
        
        if kernel == "rbf":
            pass
            # g_i = np.exp(-similarities[i]/(2*sigma_sq)) 
            # G_i = np.exp(-squareform(pdist(support_matrix, metric='sqeuclidean'))/(2*sigma_sq)) 
        elif kernel =="ip":
            g_i = np.dot(x_support, x)
            G_i = x_support @ x_support.T
            
        x_opt = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-6)
        weight_values[i] = x_opt/np.sum(x_opt)
    nnk_time = time.time() - start_time
    print(f"Neighborhood sparsity: {np.count_nonzero(weight_values>1e-6)}, Time taken: {nnk_time}")

    
    return weight_values, indices