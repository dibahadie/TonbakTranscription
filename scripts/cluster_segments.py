import os
import argparse
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from features import extract_features

def get_chunk_number(filename):
    """
    Extracts the chunk number from filenames to ensure correct numerical sorting.
    """
    try:
        base = os.path.splitext(filename)[0]
        num_str = base.split('_')[-1]
        return int(num_str)
    except ValueError:
        return -1

def main():
    parser = argparse.ArgumentParser(description="Cluster audio segments globally and save lab results.")
    parser.add_argument('-i', '--input', required=True, help='Path to the directory containing the song chunk folders.')
    parser.add_argument('-o', '--output', required=True, help='Path to the output folder to save the .raw.lab files.')
    parser.add_argument('-c', '--clusters', type=int, default=4, help='Number of clusters to form.')
    parser.add_argument('-a', '--algorithm', choices=['kmeans', 'gmm', 'agglomerative'], default='kmeans', help='Clustering algorithm to use.')
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    # Define and create the required output directory
    lab_out_dir = os.path.join(output_dir, 'seq', 'lab_phon')
    os.makedirs(lab_out_dir, exist_ok=True)

    if not os.path.isdir(parent_dir):
        print(f"Error: Input directory does not exist: {parent_dir}")
        return

    all_features = []
    all_metadata = []

    print("Phase 1: Extracting features from all folders...")
    for song_folder_name in os.listdir(parent_dir):
        song_dir = os.path.join(parent_dir, song_folder_name)
        
        if not os.path.isdir(song_dir) or song_folder_name == 'seq':
            continue
            
        # Read the intermediate lab file from output/seq/lab_phon
        seg_lab_path = os.path.join(lab_out_dir, f"{song_folder_name}.seg.lab")
        if not os.path.isfile(seg_lab_path):
            continue
            
        print(f"  Reading segments from '{song_folder_name}' via intermediate lab...")
        
        with open(seg_lab_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time, end_time, chunk_filename = float(parts[0]), float(parts[1]), parts[2]
                    wav_path = os.path.join(song_dir, chunk_filename)
                    
                    if os.path.isfile(wav_path):
                        features, _ = extract_features(wav_path)
                        
                        if features is not None:
                            all_features.append(features)
                            all_metadata.append({
                                'folder_name': song_folder_name,
                                'start': start_time,
                                'end': end_time
                            })

    total_segments = len(all_features)
    if total_segments < 4:
        print(f"Error: Only {total_segments} valid segments found across all folders. Need at least 4 to cluster.")
        return

    print(f"\nPhase 2: Normalizing and global clustering of {total_segments} segments (k={args.clusters})...")
    
    # We are back to simple 1D arrays, so standard scaling and clustering works perfectly
    X = np.array(all_features)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    if args.algorithm == 'kmeans':
        model = KMeans(n_clusters=args.clusters, random_state=42, n_init='auto')
    elif args.algorithm == 'gmm':
        model = GaussianMixture(n_components=args.clusters, random_state=42)
    elif args.algorithm == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=args.clusters)
        
    labels = model.fit_predict(X_normalized)
    cluster_labels = labels + 1

    print("\nPhase 3: Generating final .raw.lab files...")
    grouped_results = {}
    for meta, cluster_id in zip(all_metadata, cluster_labels):
        folder_name = meta['folder_name']
        if folder_name not in grouped_results:
            grouped_results[folder_name] = []
        
        grouped_results[folder_name].append({
            'start': meta['start'],
            'end': meta['end'],
            'cluster_id': cluster_id
        })

    for folder_name, segments in grouped_results.items():
        lab_filepath = os.path.join(lab_out_dir, f"{folder_name}.raw.lab")
        
        with open(lab_filepath, 'w') as lab_file:
            for seg in segments:
                # Write exact unpadded timestamps stored from the intermediate lab
                lab_file.write(f"{seg['start']:.6f} {seg['end']:.6f} {seg['cluster_id']}\n")
                
        print(f"  Saved: {lab_filepath}")

    print("\nAll done!")

if __name__ == "__main__":
    main()