from librosa.filters import get_window
from librosa import get_fftlib
from librosa import util
import numpy as np
import librosa as lb
from librosa.util.exceptions import ParameterError

MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
def expand_to(x, ndim, axes):
    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    if len(axes) != x.ndim:
        raise ParameterError(
            "Shape mismatch between axes={} and input x.shape={}".format(axes, x.shape)
        )

    if ndim < x.ndim:
        raise ParameterError(
            "Cannot expand x.shape={} to fewer dimensions ndim={}".format(x.shape, ndim)
        )

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def stft_matlab( y, n_fft=2048, hop_length=None, win_length=None, dtype=None, window="hann", pad_mode="constant"):
    
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
   
    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    #fft_window = util.pad_center(fft_window, n_fft)
    fft_window = np.pad(fft_window, (0, n_fft-win_length))

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    

    if n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too large for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = util.frame(y, frame_length=win_length, hop_length=hop_length)
    y_frames = np.pad(y_frames,[(0,n_fft-win_length),(0,0)])

    fft = get_fftlib()

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    stft_matrix = np.empty(shape, dtype=dtype, order="F")

    n_columns = MAX_MEM_BLOCK // (
        np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[-1])

        stft_matrix[..., bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix

def spectrogram_tensor_prep(y, sr, win_lengths, hop_length, n_fft, window='hann', center=True):
    win_lengths = sorted(win_lengths)
    orig_energy = np.linalg.norm(y) ** 2
    
    if center:
        s = np.abs(lb.stft(y, win_length=win_lengths[0], hop_length=hop_length, n_fft=n_fft, window=window, center=True)) ** 2
        spec_energy = np.sum(s)
        spectrograms = np.zeros([s.shape[0], s.shape[1], len(win_lengths)])
        spectrograms[:,:,0] = s*orig_energy/spec_energy
        for ind in range(1, len(win_lengths)):
            inter = np.abs(lb.stft(y, win_length=win_lengths[ind], hop_length=hop_length, n_fft=n_fft, window=window, center=True)) ** 2
            spec_energy = np.sum(inter)
            spectrograms[:,:,ind] = inter*orig_energy/spec_energy
            
    else:
        spectrogramsBeforeTrim = []
        spectrogramsBeforeTrim.append(stft_matlab(y, win_length=win_lengths[0], hop_length=hop_length, n_fft=n_fft, window=window, pad_mode='constant'))
        for ind in range(1, len(win_lengths)):
            y_shifted = np.pad(y, ((win_lengths[ind] - win_lengths[0])//2, 0))
            spectrogramsBeforeTrim.append(stft_matlab(y_shifted, win_length=win_lengths[ind], hop_length=hop_length, n_fft=n_fft, window=window, pad_mode='constant'))
            
        spec_length = spectrogramsBeforeTrim[-1].shape[1]
        spectrograms = np.zeros([n_fft//2+1, spec_length, len(win_lengths)])

        for ind in range(len(win_lengths)):
            inter = np.abs(spectrogramsBeforeTrim[ind][:, 0:spec_length]) ** 2
            spec_energy = np.sum(inter)
            spectrograms[:,:,ind] = inter*orig_energy/spec_energy


    return spectrograms

