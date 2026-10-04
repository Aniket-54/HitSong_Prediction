import os
import glob
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
import concurrent.futures

# ============================================================
# CONFIGURATION
# ============================================================
CHORUSES_DIR = "choruses"
OUTPUT_CSV = "chorus_features.csv"
SR = 22050

# 7 statistical transformations per paper methodology
def get_statistics(feature_matrix):
    """
    Takes a 2D feature matrix and
    collapses the time dimension using 7 statistics.
    Returns a 1D numpy array.
    """
    _min = np.min(feature_matrix, axis=1)
    _mean = np.mean(feature_matrix, axis=1)
    _median = np.median(feature_matrix, axis=1)
    _max = np.max(feature_matrix, axis=1)
    _std = np.std(feature_matrix, axis=1)
    _skew = skew(feature_matrix, axis=1)
    _kurtosis = kurtosis(feature_matrix, axis=1)
    
    return np.concatenate((_min, _mean, _median, _max, _std, _skew, _kurtosis))

# ============================================================
# FEATURE EXTRACTION (518 Dimensions)
# ============================================================
def extract_audio_features(file_path, label):
    """
    Extracts the 11 major features from Table 1 and applies 7 stats.
    Total dimensions: (12*3 + 20 + 1 + 1 + 1 + 7 + 1 + 6 + 1) * 7 = 518
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        
        # 1. Chroma STFT (12 dims -> 84)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # 2. Chroma CQT (12 dims -> 84)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 3. Chroma CENS (12 dims -> 84)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        
        # 4. MFCC (20 dims -> 140)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 5. RMS (1 dim -> 7)
        rms = librosa.feature.rms(y=y)
        
        # 6. Spectral Centroid (1 dim -> 7)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # 7. Spectral Bandwidth (1 dim -> 7)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # 8. Spectral Contrast (7 dims -> 49)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 9. Spectral Rolloff (1 dim -> 7)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # 10. Tonnetz (6 dims -> 42)
        # Tonnetz requires harmonic component of the signal
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        
        # 11. Zero Crossing Rate (1 dim -> 7)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Combine all base features into a list
        features = [
            chroma_stft, chroma_cqt, chroma_cens, mfcc, rms, 
            spec_cent, spec_bw, spec_contrast, spec_rolloff, tonnetz, zcr
        ]
        
        # Apply the 7 statistics to each feature and concatenate into a single flat array
        stat_features = np.concatenate([get_statistics(f) for f in features])
        
        # Append metadata
        track_name = os.path.basename(file_path).replace("_chorus.wav", "")
        row = [track_name] + stat_features.tolist() + [label]
        return row
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ============================================================
# MAIN EXECUTION
# ============================================================
def process_dataset():
    print("Generating feature matrix.")
    
    # We will label Hit as 1, Non-Hit as 0
    categories = {"hit": 1, "non_hit": 0}
    dataset = []
    
    for category, label in categories.items():
        folder_path = os.path.join(CHORUSES_DIR, category)
        wav_files = glob.glob(os.path.join(folder_path, "*_chorus.wav"))
        
        if not wav_files:
            print(f"Warning: No _chorus.wav files found in {folder_path}")
            continue
            
        print(f"Extracting features for {len(wav_files)} {category} tracks...")
        
        # Run extraction in parallel to speed things up
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(extract_audio_features, path, label): path for path in wav_files}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    dataset.append(result)

    if dataset:
        # Generate column names dynamically to match the 518 dimension count
        col_names = ["track_name"] + [f"feature_{i}" for i in range(1, 519)] + ["is_hit"]
        
        df = pd.DataFrame(dataset, columns=col_names)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Extracted {df.shape[1]-2} features across {df.shape[0]} tracks.")
        print(f"Saved to {OUTPUT_CSV}")
    else:
        print("\nFailed to extract features.")

if __name__ == "__main__":
    process_dataset()