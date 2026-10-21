import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Map a target cluster to 'tom' and remaining clusters to a broader category.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing the clustered label files.")
    parser.add_argument("--tom_cluster", type=str, required=True, help="The cluster number representing 'tom' (e.g., '0').")
    parser.add_argument("--other_label", type=str, default="edge", help="Label for all other remaining clusters (default: 'edge').")
    
    args = parser.parse_args()
    
    target_cluster = f"cluster_{args.tom_cluster}"
    processed_count = 0
    
    for filename in os.listdir(args.input_dir):
        # Skip already processed files
        if filename.endswith('.partial.lab'):
            continue
            
        # Process only standard label files
        if not (filename.endswith('.lab') or filename.endswith('.txt')):
            continue
            
        filepath = os.path.join(args.input_dir, filename)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end, label = parts[0], parts[1], parts[2]
                
                # If it matches the target cluster, label it 'tom'
                if label == target_cluster:
                    label = "tom"
                # If it is any OTHER cluster, label it as the generic edge/high category
                elif label.startswith("cluster_"):
                    label = args.other_label
                    
                new_lines.append(f"{start} {end} {label}\n")
            else:
                new_lines.append(line)
                
        # Generate the new filename (e.g., track_01.partial.lab)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.partial.lab"
        output_filepath = os.path.join(args.input_dir, output_filename)
        
        with open(output_filepath, 'w') as f:
            f.writelines(new_lines)
            
        print(f"Saved: {output_filename} (Mapped {target_cluster} -> tom, others -> {args.other_label})")
        processed_count += 1

    print(f"\nComplete! Processed {processed_count} files.")

if __name__ == "__main__":
    main()