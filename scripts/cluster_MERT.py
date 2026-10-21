import os
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

def plot_global_clusters(features_scaled, cluster_labels, output_image_path):
    """Reduces feature dimensions to 2D using t-SNE for maximum visual separation."""
    if features_scaled.shape[1] < 2 or len(features_scaled) < 2:
        print("Not enough data to generate a plot.")
        return
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    scatter_data = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scatter_data[:, 0], scatter_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=15)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("Global Tonbak Stroke Clusters (MERT + t-SNE Projection)")
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
    parser = argparse.ArgumentParser(description="Batch cluster Tonbak segments using MERT embeddings and GMM.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder containing the .wav audio files.")
    parser.add_argument("--label_dir", type=str, required=True, help="Folder containing the initial label files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save the clustered label files and global plot.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of distinct stroke types (default: 3).")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_features = []
    global_metadata = []   
    file_segments = {}     
    output_filenames = {}  
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # PHASE 1: Extract features using MERT
    # ==========================================
    model_name = "m-a-p/MERT-v1-95M"
    print(f"Loading MERT music understanding model on {device.upper()}...")
    
    # MERT requires trust_remote_code=True for its custom architecture
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Dynamically pull the model's required sampling rate (typically 24000 for MERT)
    target_sr = processor.sampling_rate
    
    print("Extracting neural embeddings across the dataset...")
    for audio_filename in os.listdir(args.audio_dir):
        if not audio_filename.endswith('.wav'):
            continue
            
        base_name = os.path.splitext(audio_filename)[0]
        audio_path = os.path.join(args.audio_dir, audio_filename)
        label_path = get_best_label_file(args.label_dir, base_name)
        
        if not label_path:
            continue
            
        output_filenames[base_name] = os.path.basename(label_path).replace('_corrected', '')
        
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
            
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        for i in seg_indices:
            seg = segments[i]
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            y_slice = y[start_sample:end_sample]
            
            # Pad extremely short segments to 50ms so MERT's CNN layers don't crash
            min_samples = int(target_sr * 0.05)
            if len(y_slice) < min_samples:
                y_slice = librosa.util.fix_length(y_slice, size=min_samples)
                
            inputs = processor(y_slice, sampling_rate=target_sr, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # MERT outputs embeddings over time: shape [batch, time_steps, 768]
            # We take the mean across the time_steps (dim=1) to get one single stroke fingerprint
            stroke_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            global_features.append(stroke_embedding)
            global_metadata.append({'base_name': base_name, 'seg_idx': i})

    # ==========================================
    # PHASE 2: Global Clustering & Plotting
    # ==========================================
    features_array = np.array(global_features)
    
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