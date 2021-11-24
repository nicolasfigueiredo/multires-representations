import numpy as np
import librosa as lb
import scipy.interpolate as inter
from .spectrogram_tensor import spectrogram_tensor_prep
from .util import compress_dB_norm, structure_tensor
from .stfcht import fanchirpogram

def mrfci(x, sr, win_lengths,FChT_flags, asym_flags, Nf_structure_tensor, hop_length, N_alphas, C_limits, R,sigma_t, sigma_f, center=True, interpolate=False, window='hann', pad_mode='reflect'):

    sigma_m = sigma_t*sr / hop_length
    sigma_k = sigma_f*Nf_structure_tensor / sr
    Ngauss = 3*max([sigma_m, sigma_k])

    freq_struct_tens = lb.fft_frequencies(sr=sr, n_fft=win_lengths[0])
### PAD MODE ####    
    STFT = lb.stft(x, hop_length=hop_length, window=window, n_fft=win_lengths[0], center=center, pad_mode=pad_mode)


# Mudei o Tempo para alinhamento a esquerda
#time_struct_tens = lb.frames_to_time(range(x.shape[0]//hop + 1), sr=fs, hop_length=hop)
    time_struct_tens = lb.times_like(STFT, sr=sr, hop_length=hop_length, n_fft=win_lengths[0])

    X_structure_tensor = compress_dB_norm(abs(STFT) ** 2, R)
    C, v1, v2 = structure_tensor(X_structure_tensor, Ngauss, sigma_m, sigma_k)
    C = C/np.max(C)

    k_array = np.arange(freq_struct_tens.shape[0])
    alphas = (v2/v1) * sr / (hop_length* np.tile(k_array, (time_struct_tens.shape[0], 1)).T)
    alphas[0,:] = -1e11 # pq um numero negativo?
    alphas[np.isnan(alphas)] = 0 # pq zero na divisão por zero? numero grande não seria melhor?

    freq_combined = lb.fft_frequencies(sr=sr, n_fft=win_lengths[-1])
    if center:
        alphas_interp = inter.interp1d(freq_struct_tens, alphas, kind='linear', axis=0, assume_sorted=True)(freq_combined)
        C_interp = inter.interp1d(freq_struct_tens, C, kind='linear', axis=0, assume_sorted=True)(freq_combined)
    else:
        time_combined = np.arange(win_lengths[-1]/(2*sr), ((np.floor((x.shape[0] - win_lengths[-1]) / hop_length) + 1) * hop_length + win_lengths[-1]/2 -1)/sr, hop_length/sr )
        m_grid_orig, k_grid_orig  = np.meshgrid(time_struct_tens, freq_struct_tens)
        m_grid_final, k_grid_final = np.meshgrid(time_combined, freq_combined)
        alphas_interp = inter.griddata( (m_grid_orig.flatten(), k_grid_orig.flatten()), alphas.flatten(), (m_grid_final, k_grid_final) , method='linear', fill_value=0)
        C_interp = inter.griddata( (m_grid_orig.flatten(), k_grid_orig.flatten()), C.flatten(), (m_grid_final, k_grid_final) , method='linear', fill_value=0)

    alpha_max = 2*sr/win_lengths[-1]
    alphas_ind = np.arange(N_alphas+1)
    initial_alphas = np.tan(alphas_ind * np.arctan(alpha_max)/N_alphas)
    initial_alphas = np.append(-initial_alphas[:0:-1], initial_alphas)
    ## tb deve ter um jeito mehor de fazer isso
    if len(C_limits.shape) == 1:
        C_limits = np.vstack([C_limits, np.arange(1, len(C_limits)+1)])

    STFT_tensor = spectrogram_tensor_prep(x, sr, win_lengths, hop_length, window=window, center=center, interpolate=True, interp_kind='linear')
    TFRs_tensor = np.empty([STFT_tensor.shape[0], STFT_tensor.shape[1], len(initial_alphas), STFT_tensor.shape[2]], dtype='complex128')
    for ind_Nf in range(len(win_lengths)): # aqui ele repete o laço para primeiro Nf
        if FChT_flags[ind_Nf]: # vai ter intâncias da Fanchirp
            for alphas_ind in range(1, len(initial_alphas)-1):
                if initial_alphas[alphas_ind] != 0:
                    STFChT= fanchirpogram(x, initial_alphas[alphas_ind], sr, n_fft=win_lengths[ind_Nf], hop_length=hop_length, win_length=win_lengths[ind_Nf], window=window, center=center, pad_mode=pad_mode)
                    if center:
                        STFChT_interp = inter.interp1d(lb.fft_frequencies(sr=sr, n_fft=win_lengths[ind_Nf]),
                                                    np.abs(STFChT), kind='linear', axis=0, assume_sorted=True)(lb.fft_frequencies(sr=sr, n_fft=win_lengths[-1]))
                    else:
                        freq_struct_tens = lb.fft_frequencies(sr=sr, n_fft=win_lengths[ind_Nf])
                        time_struct_tens = lb.times_like(STFChT, sr=sr, hop_length=hop_length, n_fft=win_lengths[ind_Nf])
                        m_grid_orig, k_grid_orig  = np.meshgrid(time_struct_tens, freq_struct_tens)
                        STFChT_interp = inter.griddata((m_grid_orig.flatten(), k_grid_orig.flatten()), np.abs(STFChT).flatten(), (m_grid_final, k_grid_final) , method='linear', fill_value=0)
                    TFRs_tensor[:, :, alphas_ind, ind_Nf] = STFT_tensor[:,:,ind_Nf]
                else:
                    TFRs_tensor[:, :, alphas_ind, ind_Nf] = STFT_tensor[:,:,ind_Nf]
            #STFT is used as first and last Layers ->> PQ?
            TFRs_tensor[:, :, 0, ind_Nf] = STFT_tensor[:,:,0] # pq o short interp e não o interp?
            TFRs_tensor[:, :, -1, ind_Nf] = STFT_tensor[:,:,0]
        else: # TFR vai ser a DFT em tudo
            for alphas_ind in range(len(initial_alphas)):
                TFRs_tensor[:, :, alphas_ind, ind_Nf] = STFT_tensor[:,:,ind_Nf]

    TFR_comb = np.zeros(TFRs_tensor.shape[:-2], dtype='complex128')
    weights_tensor_alphas = np.zeros(TFRs_tensor.shape[:-1], dtype='complex128')

    for ind_alphas in range(TFRs_tensor.shape[2] -1):
        aux_alphas_matrix1 = np.zeros_like(TFR_comb)
        aux_alphas_matrix2 = np.zeros_like(TFR_comb)

        inds_matrix = np.logical_and(alphas_interp > initial_alphas[ind_alphas] , alphas_interp <= initial_alphas[ind_alphas + 1])
        
        alpha_values = alphas_interp[inds_matrix]
        
        weight_alpha_2  = (alpha_values - initial_alphas[ind_alphas]) / (initial_alphas[ind_alphas +1] - initial_alphas[ind_alphas])
        weight_alpha_1 = 1 - weight_alpha_2
        aux_alphas_matrix1[inds_matrix] = weight_alpha_1
        aux_alphas_matrix2[inds_matrix] = weight_alpha_2

        weights_tensor_alphas[:,:, ind_alphas] = aux_alphas_matrix1 + weights_tensor_alphas[:, :, ind_alphas]
        weights_tensor_alphas[:, :, ind_alphas + 1] = aux_alphas_matrix2 + weights_tensor_alphas[:, :, ind_alphas + 1]


    for ind_C in range(C_limits.shape[1] -1):
        C_matrix1 = np.zeros_like(TFR_comb)
        C_matrix2 = np.zeros_like(TFR_comb)

        inds_matrix  = np.logical_and(C_interp > C_limits[0, ind_C], C_interp <= C_limits[0, ind_C  + 1])

        C_values = C_interp[inds_matrix]

        weight_C_2 = (C_values - C_limits[0, ind_C])/(C_limits[0, ind_C + 1] - C_limits[0, ind_C])
        weight_C_1 = 1 - weight_C_2

        C_matrix1[inds_matrix] = weight_C_1
        C_matrix2[inds_matrix] = weight_C_2

        aux_matrix_1 = C_matrix1 * np.sum(weights_tensor_alphas * TFRs_tensor[:,:,:, int(C_limits[1, ind_C])-1], axis=2)
        aux_matrix_2 = C_matrix2 * np.sum(weights_tensor_alphas * TFRs_tensor[:,:,:, int(C_limits[1, ind_C + 1])-1], axis=2)

        TFR_comb[inds_matrix] = aux_matrix_1[inds_matrix] + aux_matrix_2[inds_matrix]

    # Preenchendo TFR onde C <= C_limits[0]
    TFR_comb[C_interp <= C_limits[0,0]] = STFT_tensor[:,:,0][C_interp <= C_limits[0,0]]
    # Preenchendo TFR onde C > C_limits[-1]
    aux_matrix = np.sum(weights_tensor_alphas * TFRs_tensor[:,:,:,-1], axis=2)
    TFR_comb[C_interp > C_limits[0, -1]] = aux_matrix[C_interp > C_limits[0,-1]]
    # preenchendo TFR nos ataques
    TFR_comb[alphas_interp < initial_alphas[0]] = STFT_tensor[:,:,0][alphas_interp <= initial_alphas[0]]
    TFR_comb[alphas_interp >= initial_alphas[-1]] = STFT_tensor[:,:,0][alphas_interp >= initial_alphas[-1]]

    return TFR_comb, STFT_tensor#, Time_combined, Freq_combined
