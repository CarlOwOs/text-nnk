import time
import torch

from utils.FAISS_Search import FAISS_Search
from utils.approximate_nnk_solver import approximate_nnk

def nnk_graph(features, top_k, cuda_available, device):
    
    n, d = features.shape
    print(n, d)

    features = features.to(device)
    support_data = features

    start_time = time.time()
    faiss_search = FAISS_Search(dim=d, use_gpu=cuda_available)
    faiss_search.add(support_data)

    similarities, indices = faiss_search.search(features, top_k=top_k+1)

    similarities = similarities[:, 1:]
    indices = indices[:, 1:]

    sigma_sq = 0.5 * torch.mean(similarities[:, 1:]).item() ** 2

    weight_values = torch.zeros((n, top_k), device=device)
    error = torch.zeros((n, 1), device=device)

    for i in range(n):
        neighbor_indices = indices[i, :]
        support_matrix = support_data[neighbor_indices]
        
        #sigma_sq = (similarities[i, 14]/3)**2    
        #sigma_sq = np.quantile(similarities[i, :], 1)**2
        g_i = torch.exp(-similarities[i]/(2*sigma_sq)) # rbf
        print("g_i", g_i.shape)
        G_i = torch.exp(-torch.cdist(support_matrix, support_matrix, p=2)/(2*sigma_sq)) # rbf
        print("G_i", G_i.shape)
        weight_values[i], error[i] = approximate_nnk(G_i, g_i, g_i,
                                            x_tol=1e-6,
                                            num_iter=100, eta=0.05)
    nnk_time = time.time() - start_time
    print(f"Neighborhood sparsity: {torch.count_nonzero(weight_values>1e-6)}, Time taken: {nnk_time}")

    
    return weight_values, error