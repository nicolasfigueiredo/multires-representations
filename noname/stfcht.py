import numpy as np
import math as m
import scipy.signal as sig
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


def stfcht(x, alphas, win_length=1024, w='hanning', hop_size=256, sr=48000):
    # Arrumar isso
    w = np.hanning(win_length)
    K = len(w)

    time = np.arange(K/(2*sr),
                     ((np.floor((len(x) - K) / hop_size) + 1) * hop_size + K / 2 - 1) / sr,
                     hop_size/sr)
    freq = np.arange(0, K // 2 + 1) * sr / K

    if len(alphas) < len(time):
        alpha = alphas[0]
        M = 2 ** (m.ceil(np.log2(m.ceil(K / (1-abs(alpha/sr) * K/2)))))
        if M > K:
            # porque não hanning(M)? os resamples não são iguais (como faz?)
            w = sig.resample(w, M)

        if alpha != 0:
            tn = (np.arange(0, K) - (K - 1) / 2) / sr
            tr = phi_warp(tn[0], alpha) + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M
            tt = psi_warp(tr, alpha)
        else:
            tn = (np.arange(0, K) - (K - 1) / 2) / sr
            tt = tn[0] + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M

        # da pra usar M no lugar de len(w)
        frames = np.zeros((len(w), len(time)))
        for frame_ind in range(len(time + 1)):
            t0 = frame_ind * hop_size
            xn = x[t0:t0+K]
            xr = interp1d(tn, xn)(tt)
            frames[:, frame_ind] = xr * w
            X = np.fft.fft(frames, axis=1)
            STFChT = X[1:K // 2 + 1, :]
    else:
        for frame_ind in range(len(time)):
            alpha_max = max(alphas)
            M = 2 ** np.ceil(np.log2(np.ceil(K / (1-abs(alpha_max/sr) * K/2))))
            w = sig.resample(w, M, K)  # Era N troquei por K

            if alphas[frame_ind] != 0:
                tn = (np.arange(0, K) - (K - 1) / 2) / sr
                tr = phi_warp(tn[0], alphas[frame_ind]) + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M
                tt = psi_warp(tr, alphas[frame_ind])
            else:
                tn = (np.arange(0, K) - (K - 1) / 2) / sr
                tt = tn[0] + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M

            t0 = frame_ind * hop_size
            xn = x[t0:t0+K]
            xr = interp1d(tn, xn)(tt)
            Xr = np.fft.fft(xr * w)
            STFChT[:, frame_ind] = Xr[0:K // 2 + 1]

    return STFChT, freq, time
