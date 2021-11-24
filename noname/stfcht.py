import numpy as np
from math import ceil
from scipy.interpolate import interp1d
from librosa.filters import get_window
from librosa.util import  dtype_r2c, MAX_MEM_BLOCK, frame

def psi_warp(x, alfa):
    return (-1+np.sqrt(1+2*alfa*x))/alfa

def phi_warp(x, alfa):
    return (1+ 0.5 * alfa * x ) * x

def fanchirpogram(y, alpha, sr, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', dtype=None):
    
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    M = 1 << (abs(ceil(win_length/ (1-abs(alpha/sr) * win_length/2)) -1)).bit_length()
    
    fft_window = get_window(window, max(win_length, M), fftbins=True)

    # Pad the window out to n_fft size
    #fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)
    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)
    
    if alpha != 0:
        tn = (np.arange(win_length)-(win_length-1)/2)/sr
        tr = phi_warp(tn[0], alpha) + (np.arange(M) + 1/2 )*(tn[-1]-tn[0])/M
        tt = psi_warp(tr, alpha)
    else: # caso alpha = 0 não é a fft normal?
        tn = (np.arange(0, win_length) - (win_length - 1) / 2) / sr                           
        tt = tn[0] + (np.arange(0, M) + 1 / 2) * (tn[-1] - tn[0]) / M

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = np.fft.rfft(
            fft_window * interp1d(tn, y_frames[:, bl_s:bl_t], axis=0)(tt), axis=0
        )[:win_length//2+1, :]
    return stft_matrix