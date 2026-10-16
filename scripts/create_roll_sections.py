import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract roll (riz) and hit intervals from clustered lab files.")
    parser.add_argument('-i', '--input', required=True, help='Path to the main input directory containing the "seq" folder.')
    parser.add_argument('-c', '--cluster', required=True, type=str, help='The cluster number that represents a riz section.')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.input)
    lab_in_dir = os.path.join(base_dir, 'seq', 'lab_phon')
    lab_out_dir = os.path.join(base_dir, 'seq', 'lab_roll')

    if not os.path.isdir(lab_in_dir):
        print(f"Error: Input lab directory not found at {lab_in_dir}")
        return

    os.makedirs(lab_out_dir, exist_ok=True)

    processed_files = 0
    
    for filename in os.listdir(lab_in_dir):
        if not filename.endswith('.raw.lab'):
            continue
            
        in_filepath = os.path.join(lab_in_dir, filename)
        song_name = filename.replace('.raw.lab', '')
        out_filepath = os.path.join(lab_out_dir, f"{song_name}.roll.lab")
        
        intervals = []
        current_start = None
        current_end = None
        current_type = None
        
        with open(in_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                    
                start = float(parts[0])
                end = float(parts[1])
                cluster_id = parts[2]
                
                # Determine interval type based on the cluster
                seg_type = "roll" if cluster_id == args.cluster else "hit"
                
                if current_type is None:
                    # Initialize the first interval
                    current_type = seg_type
                    current_start = start
                    current_end = end
                elif seg_type == current_type:
                    # Extend the current interval
                    current_end = end
                else:
                    # Type changed: save the previous interval and start a new one
                    intervals.append((current_start, current_end, current_type))
                    current_type = seg_type
                    current_start = start
                    current_end = end
                        
        # Append the final tracked interval
        if current_type is not None:
            intervals.append((current_start, current_end, current_type))
            
        # Write the resulting merged boundaries
        with open(out_filepath, 'w') as f:
            for start, end, itype in intervals:
                f.write(f"{start:.6f} {end:.6f} {itype}\n")
                
        processed_files += 1
        print(f"  Saved {len(intervals)} intervals to {out_filepath}")

    print(f"\nDone. Processed {processed_files} files.")

if __name__ == "__main__":
    main()