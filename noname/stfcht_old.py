import numpy as np
from math import ceil
from scipy.io import loadmat
from scipy.interpolate import interp1d


def psi_warp(x, alpha):
    """
    Function description
    """
    return (-1 + np.sqrt(1 + 2 * alpha * x)) / alpha


def phi_warp(x, alpha):
    """
    Function description
    """
    return (1 + 1/2 * alpha * x) * x


def fanchirpogram(x, w, hop_length, sr, alphas):
    
    K = len(w)
    
    time = np.arange(K/(2*sr), ((np.floor((x.shape[0] - K) / hop_length) + 1) * hop_length + K/2 -1)/sr, hop_length/sr)
    freq = np.arange(K/2 + 1) * sr / K
    
    if isinstance(alphas, (float, int, np.float64)):
        alpha_max = alphas
        alphas = np.full_like(time, alpha_max)
    elif len(alphas) < len(time):
        alpha_max = alphas[0]
        alphas = np.full_like(time, alpha_max)
    else:
        alpha_max = max(alphas)
    M = 1 << (abs(ceil(K / (1-abs(alpha_max/sr) * K/2)) -1)).bit_length()
    if M > K:
        if K == 2048:
            ref2 = loadmat('fanchirp_mrfci.mat')
            w1 = ref2['w1'].squeeze() # a janela interpolada faz diferença Muita DIFERENÇA n Fanchirp
        elif K == 4096:
            ref2 = loadmat('fanchirp_mrfci2.mat')
            w1 = ref2['w1'].squeeze() # a janela interpolada faz diferença Muita DIFERENÇA na Fanchirp
        else:
            #w1 = sig.resample(w, M)
            w1 = np.hanning(M+2)[1:-1]
    else:
        w1 = w

    
    frames = np.zeros((len(w1), len(time))) 
    for frame_ind in range(len(time)):   
        if alphas[frame_ind] != 0:
            tn = (np.arange(K)-(K-1)/2)/sr
            tr = phi_warp(tn[0], alphas[frame_ind]) + (np.arange(M) + 1/2 )*(tn[-1]-tn[0])/M
            tt = psi_warp(tr, alphas[frame_ind])
        else: # caso alpha = 0 não é a fft normal?
            tn = (np.arange(0, K) - (K - 1) / 2) / sr                           
            tt = tn[0] + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M
        
        t0 = frame_ind * hop_length
        xn = x[t0: t0 + K]
        xr = interp1d(tn, xn)(tt)
        frames[:, frame_ind] = xr * w1 #
        
    X = np.fft.fft(frames, axis=0)
    return X[0 : K//2+1, :], freq, time