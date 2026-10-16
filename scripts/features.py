import warnings
import numpy as np
import librosa
import scipy.stats

def extract_features(file_path):
    """
    Extracts features explicitly designed to separate rolls (noise/friction)
    from fast individual hits (tonal/spiky) using statistical moments and spectral flatness.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=None)
            
        if len(y) == 0:
            return None, 0.0
            
        duration = len(y) / sr
        window_size = min(2048, len(y))
        hop_size = max(1, window_size // 4)
        
        rms = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_size)[0]
        if len(rms) == 0:
            return None, 0.0
            
        # 1. Statistical Shape (Spikiness vs. Noise)
        # Fast hits have extreme spikes (high kurtosis) and lean heavily to one side (high skew).
        # Rolls behave more like continuous random noise (lower kurtosis, near-zero skew).
        sig_kurtosis = scipy.stats.kurtosis(y)
        sig_skew = scipy.stats.skew(np.abs(y))
        
        # 2. Spectral Flatness (Friction vs. Resonance)
        # Rolls are noisy (closer to 1). Hits ring the drum's tonal frequencies (closer to 0).
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=window_size, hop_length=hop_size)[0]
        mean_flatness = np.mean(flatness)
        
        # 3. Crest Factor & Valley Depth
        crest_factor = np.max(rms) / (np.mean(rms) + 1e-6)
        min_max_ratio = np.min(rms) / (np.max(rms) + 1e-6)
        
        # 4. Spectral Centroid (Brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=window_size, hop_length=hop_size)[0]
        mean_centroid = np.mean(cent)
        
        # 5. Energy Distribution
        quarter_idx = max(1, len(rms) // 4)
        front_energy_ratio = np.sum(rms[:quarter_idx]) / (np.sum(rms) + 1e-6)

        features = np.array([
            sig_kurtosis,
            sig_skew,
            mean_flatness,
            crest_factor,
            min_max_ratio,
            mean_centroid,
            front_energy_ratio
        ])
        
        return features, duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0.0