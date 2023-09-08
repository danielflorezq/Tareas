import numpy as np

class SVD:
    def __init__(self, k):
        self.k = k

    def transform(self, X):
        U, S, VT = np.linalg.svd(X)
        U_truncated = U[:, :self.k]
        S_truncated = np.diag(S[:self.k])
        VT_truncated = VT[:self.k, :]
        return U_truncated, S_truncated, VT_truncated
