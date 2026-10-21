import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract only the text transcription from Audacity-style label files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing the final .lab or .txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save the clean transcription text files.")
    parser.add_argument("--separator", type=str, default=" ", help="Character to separate strokes. Default is a space. (Use '\\n' for new lines).")
    
    args = parser.parse_args()
    
    # Handle the newline escape character from command line arguments safely
    separator = '\n' if args.separator == '\\n' else args.separator
    
    os.makedirs(args.output_dir, exist_ok=True)
    processed_count = 0
    
    for filename in os.listdir(args.input_dir):
        # We want to process standard label files and our partial files
        if not (filename.endswith('.partial.lab')) or filename.endswith('.seg.partial.lab'):
            continue
            
        input_filepath = os.path.join(args.input_dir, filename)
        
        with open(input_filepath, 'r') as f:
            lines = f.readlines()
            
        transcription = []
        for line in lines:
            parts = line.strip().split()
            # Audacity label formats usually contain: start_time end_time label
            if len(parts) >= 3:
                # Extract just the label (the 3rd item)
                label = parts[2]
                transcription.append(label)
                
        # Skip empty files
        if not transcription:
            continue
            
        # Create output filename (change extension to .txt)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_filepath = os.path.join(args.output_dir, output_filename)
        
        # Write the clean transcription to the new file
        with open(output_filepath, 'w') as f:
            f.write(separator.join(transcription))
            
        print(f"Exported transcription: {output_filename}")
        processed_count += 1

    print(f"\nComplete! Extracted plain text transcriptions for {processed_count} files.")

if __name__ == "__main__":
    main()