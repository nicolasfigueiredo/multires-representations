import numpy as np
import librosa as lb
import scipy.interpolate as inter


def spectrogram_tensor_prep(y, sr, win_lengths, hop_length, window='hann', center=True, interpolate=False, interp_kind='linear'):
    # np.stack([a,b,c]) para alterar a ordem
    win_lengths = sorted(win_lengths)
    original_energy = np.linalg.norm(y)
    
    if center:
        if interpolate:
            freq_final = lb.fft_frequencies(sr=sr, n_fft=win_lengths[-1])
            tensor = np.dstack([
                inter.interp1d(
                    lb.fft_frequencies(sr=sr, n_fft=win_length),
                    lb.stft(y, win_length=win_length, hop_length=hop_length, n_fft=win_length,window=window),
                    kind=interp_kind, axis=0, assume_sorted=True)(freq_final) for win_length in win_lengths[:-1]])
            tensor = np.dstack([tensor, 
                                lb.stft(y, win_length=win_lengths[-1], hop_length=hop_length, 
                                        n_fft=win_lengths[-1],window=window)])
        
        else:
            tensor = np.dstack([lb.stft(y, win_length=win_length, hop_length=hop_length, n_fft=win_lengths[-1],
                                        window=window) for win_length in win_lengths])
        
    
    else:
        tensor = lb.stft(y, win_length=win_lengths[-1], hop_length=hop_length, 
                         n_fft=win_lengths[-1], window=window, center=False, pad_mode='constant')
        time_final = lb.times_like(tensor, sr=sr, hop_length=hop_length, n_fft=win_lengths[-1])
        freq_final = lb.fft_frequencies(sr=sr, n_fft=win_lengths[-1])
        m_grid_final, k_grid_final = np.meshgrid(time_final, freq_final)
        for  win_length in win_lengths[-2::-1]:
            s = lb.stft(y, win_length=win_length, hop_length=hop_length, n_fft=win_length,
                        window=window, center=False, pad_mode='constant')
            time = lb.times_like(s, sr=sr, hop_length=hop_length, n_fft=win_length)
            freq = lb.fft_frequencies(sr=sr, n_fft=win_length)
            m_grid, k_grid = np.meshgrid(time, freq)
            rrr= inter.griddata((m_grid.flatten(), k_grid.flatten()), s.flatten(),
                    (m_grid_final, k_grid_final) , method='linear', fill_value=0)
            tensor = np.dstack([
                rrr,
                tensor])
            
    return (original_energy / np.linalg.norm(tensor, axis=(0,1), keepdims=True)) * tensor