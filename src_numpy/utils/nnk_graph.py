import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

from utils.FAISS_Search import FAISS_Search
from utils.nnk_solver import non_negative_qpsolver

def nnk_graph(features, top_k, kernel):
    
    n, d = features.shape
    print(n, d)

    features = features
    support_data = features

    start_time = time.time()
    faiss_search = FAISS_Search(dim=d, kernel=kernel, use_gpu=False)
    faiss_search.add(support_data)

    similarities, indices = faiss_search.search(features, top_k=top_k+1)
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]

    sigma_sq = 0.5 * np.mean(similarities[:, 1:]) ** 2
    
    weight_values = np.zeros((n, top_k))

    for i in range(n):
        neighbor_indices = indices[i, :]
        support_matrix = faiss_search.get_support()[neighbor_indices]
        
        #sigma_sq = (similarities[i, 14]/3)**2    
        #sigma_sq = np.quantile(similarities[i, :], 1)**2
        if kernel == "rbf":
            g_i = np.exp(-similarities[i]/(2*sigma_sq)) 
            G_i = np.exp(-squareform(pdist(support_matrix, metric='sqeuclidean'))/(2*sigma_sq)) 
        elif kernel =="ip":
            g_i = similarities[i]
            G_i = support_matrix @ support_matrix.T
            
        weight_values[i] = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-6)
    nnk_time = time.time() - start_time
    print(f"Neighborhood sparsity: {np.count_nonzero(weight_values>1e-6)}, Time taken: {nnk_time}")

    
    return weight_values