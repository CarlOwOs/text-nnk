import faiss
import faiss.contrib.torch_utils

class FAISS_Search:
    def __init__(self, dim, use_gpu=False):
        self.matrix = None
        if use_gpu:
            res = faiss.StandardGpuResources()
            # # res.noTempMemory()
            self.index = faiss.GpuIndexFlatL2(res, dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

    @staticmethod
    def _faiss_preprocess(X):
        return X.contiguous()

    def add(self, matrix):
        self.matrix = self._faiss_preprocess(matrix)
        self.index.add(self.matrix)

    def search(self, queries, top_k):
        queries = self._faiss_preprocess(queries)
        similarities, indices = self.index.search(queries, top_k)
        return similarities, indices

    def get_support(self):
        return self.matrix

    def reset(self):
        self.index.reset()
        del self.matrix