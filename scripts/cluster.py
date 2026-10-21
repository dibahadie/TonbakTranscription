import os
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

def plot_global_clusters(features_scaled, cluster_labels, output_image_path):
    """Reduces feature dimensions to 2D using t-SNE for maximum visual separation."""
    if features_scaled.shape[1] < 2 or len(features_scaled) < 2:
        print("Not enough data to generate a plot.")
        return
        
    # t-SNE actively pushes dissimilar clusters apart
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    scatter_data = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scatter_data[:, 0], scatter_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=15)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("Global Tonbak Stroke Clusters (t-SNE Projection)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    
    plt.savefig(output_image_path, dpi=150)
    plt.close()

def get_best_label_file(label_dir, base_name):
    """Finds the appropriate label file, prioritizing '_corrected' versions."""
    priorities = [
        f"{base_name}_corrected.lab",
        f"{base_name}_corrected.txt",
        f"{base_name}.lab",
        f"{base_name}.txt"
    ]
    
    for filename in priorities:
        path = os.path.join(label_dir, filename)
        if os.path.exists(path):
            return path
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch cluster unknown Tonbak segments across an entire dataset.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder containing the .wav audio files.")
    parser.add_argument("--label_dir", type=str, required=True, help="Folder containing the initial label files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save the clustered label files and global plot.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of distinct stroke types (default: 3).")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_features = []
    global_metadata = []   # To track which feature belongs to which file and segment
    file_segments = {}     # Store all segments in memory so we can save them at the end
    output_filenames = {}  # Store the target output filename for each base_name
    
    # ==========================================
    # PHASE 1: Extract features from ALL files
    # ==========================================
    print("Extracting features across the dataset...")
    for audio_filename in os.listdir(args.audio_dir):
        if not audio_filename.endswith('.wav'):
            continue
            
        base_name = os.path.splitext(audio_filename)[0]
        audio_path = os.path.join(args.audio_dir, audio_filename)
        label_path = get_best_label_file(args.label_dir, base_name)
        
        if not label_path:
            continue
            
        output_filenames[base_name] = os.path.basename(label_path).replace('_corrected', '')
        
        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
                        
        segments = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                segments.append({
                    'start': float(parts[0]),
                    'end': float(parts[1]),
                    'label': parts[2]
                })
        
        file_segments[base_name] = segments
        seg_indices = [i for i, seg in enumerate(segments) if seg['label'] == 'seg']
        
        if not seg_indices:
            continue
            
        # Load audio and extract
        y, sr = librosa.load(audio_path, sr=None)
        
        # ---> THIS LOOP WAS BROKEN IN YOUR SCRIPT <---
        for i in seg_indices:
            seg = segments[i]
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            y_slice = y[start_sample:end_sample]
            
            if len(y_slice) == 0:
                continue
                
            # 1. FIX: Normalize the volume so clustering relies entirely on timbre
            y_slice = librosa.util.normalize(y_slice)
                
            # 2. FIX: Isolate just the first 50 milliseconds (the attack/onset)
            attack_samples = int(sr * 0.05) 
            y_attack = y_slice[:attack_samples]
            
            # Zero-pad if the attack slice is shorter than librosa's FFT window
            if len(y_attack) < 2048:
                y_attack = librosa.util.fix_length(y_attack, size=2048)
                
            # 3. Extract features ONLY from the attack, not the whole slice
            # (Replace all y_slice references below with y_attack)
            
            mfccs = librosa.feature.mfcc(y=y_attack, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1) 

            time_frames = mfccs.shape[1]
            if time_frames >= 3:
                width = min(9, time_frames if time_frames % 2 != 0 else time_frames - 1)
                mfcc_delta = librosa.feature.delta(mfccs, width=width)
                mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            else:
                mfcc_delta_mean = np.zeros(mfccs.shape[0])

            cent = librosa.feature.spectral_centroid(y=y_attack, sr=sr)
            cent_mean = np.mean(cent)
            cent_std = np.std(cent)

            zcr = librosa.feature.zero_crossing_rate(y=y_attack)
            zcr_mean = np.mean(zcr)

            rolloff = librosa.feature.spectral_rolloff(y=y_attack, sr=sr, roll_percent=0.85)
            rolloff_mean = np.mean(rolloff)
            

            feature_vector = np.hstack([
                mfccs_mean, mfccs_std, mfcc_delta_mean, 
                cent_mean, cent_std, zcr_mean, rolloff_mean
            ])
            global_features.append(feature_vector)
            
            global_metadata.append({'base_name': base_name, 'seg_idx': i})

    # ==========================================
    # PHASE 2: Global Clustering & Plotting
    # ==========================================
    features_array = np.array(global_features)
    
    # Safeguard against tiny/empty datasets
    if len(features_array) == 0:
        print("No valid 'seg' segments found across the dataset. Exiting.")
        return
    elif len(features_array) == 1:
        print("Only found 1 'seg' sample. Not enough to cluster. Assigning to cluster_0.")
        global_cluster_labels = [0]
    else:
        print(f"\nClustering {len(features_array)} total segments into {args.n_clusters} clusters...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        actual_clusters = min(args.n_clusters, len(features_array))
        
        # Swap KMeans for GMM with 'full' covariance to allow elliptical clusters
        gmm = GaussianMixture(n_components=actual_clusters, covariance_type='full', random_state=42, n_init=5)
        global_cluster_labels = gmm.fit_predict(features_scaled)
        
        plot_path = os.path.join(args.output_dir, "global_clusters.png")
        plot_global_clusters(features_scaled, global_cluster_labels, plot_path)
        print(f"Global plot saved to {plot_path}")

    # ==========================================
    # PHASE 3: Update Labels and Save Files
    # ==========================================
    print("Applying global clusters and saving label files...")
    
    for meta, cluster_id in zip(global_metadata, global_cluster_labels):
        base_name = meta['base_name']
        seg_idx = meta['seg_idx']
        file_segments[base_name][seg_idx]['label'] = f"cluster_{cluster_id}"
        
    for base_name, segments in file_segments.items():
        output_path = os.path.join(args.output_dir, output_filenames[base_name])
        with open(output_path, 'w') as f:
            for seg in segments:
                f.write(f"{seg['start']:.6f} {seg['end']:.6f} {seg['label']}\n")

    print(f"\nBatch processing complete. All files saved to {args.output_dir}")

if __name__ == "__main__":
    main()