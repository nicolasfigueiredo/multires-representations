import numpy as np
import scipy.signal as sig

#function X_comp = compress_dB_norm(X, range)
def compress_dB_norm(X, r):
    X_comp = X/np.max(X)
    X_comp = 10/r*np.log10(X_comp) + 1
    X_comp[X_comp < 0] = 0
    return X_comp

def assym_gauss2D(Ngauss, sigma, assym_factor):
    N = int(np.ceil(Ngauss))
    if not N % 2:
        N -= 1
    x = np.arange(-(N-1)//2, (N+1)//2)
    gm = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2) * np.exp(-x ** 2 / (2 * sigma[0] ** 2))
    gm_2 = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2) * np.exp(-x ** 2 / (2 * (sigma[0]*assym_factor) ** 2))
    
    #gm = [gm_2(1:(end-1)/2); gm((end-1)/2+1:end)];
    gm_2[N//2:] = gm[N//2:]
    
    gk = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2) * np.exp(-x **2 / (2 * sigma[1]**2))
    
    return  np.dot(gk.reshape((N,1)), gm_2.reshape((1,N)))

#código original chama hanning mas usa hamming do matlab, verificar
# tem que ser hanning dentro
def asym_hanning(n1, n2):
    w1 = np.hamming(n1)
    w2 = np.hamming(n2)
    return np.pad(np.append(w1[:n1//2], w2[n2//2:]), (0, (n1-n2)//2))

def structure_tensor(X, Ngauss, sigma_m, sigma_k):
    D = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # Que tipo de convolução usar? R: esse é o default da implementação do imgradientxy do MATLAB
    X_m = sig.convolve2d(X, D, mode='same', boundary='symm')
    X_k = sig.convolve2d(X, D.T, mode='same', boundary='symm')
    
    G = assym_gauss2D(Ngauss, [sigma_m, sigma_k], 1) # Using symmetric ??
    X_mm = sig.convolve2d(X_m * X_m, G, 'same')
    X_mk = X_km = sig.convolve2d(X_k * X_m, G, 'same')
    X_kk = sig.convolve2d(X_k * X_k, G, 'same')
    
    C = np.zeros_like(X, dtype='float64')
    v1 = np.zeros_like(X, dtype='float64')
    v2 = np.zeros_like(X, dtype='float64')
    v12 = np.zeros_like(X, dtype='float64')
    v22 = np.zeros_like(X, dtype='float64')
    
    for m in range(X.shape[1]):
        for k in range(X.shape[0]):
            if X[k,m] > 0:
                
                T = np.array([[X_mm[k,m], X_km[k,m]],[X_mk[k,m], X_kk[k,m]]])
                e_val, e_vec = np.linalg.eig(T)
                s = np.argsort(e_val)
                val = e_val[s]
                vec = e_vec[:,s]
                # Angle of orientation
                v1[k,m] = vec[0,0]
                v2[k,m] = vec[1,0]
                
                C[k,m] = ((val[1] - val[0]) / (val[0] + val[1])) ** 2
       
    G_C = assym_gauss2D(5, [1, 1], 1) # Using symmetric
    C = sig.convolve2d(C, G_C, 'same')
    return C, v1, v2