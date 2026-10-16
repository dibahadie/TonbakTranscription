import os
import argparse
from scipy.io import wavfile

def parse_lab_file(lab_path):
    """
    Parses the lab file.
    For each line, extracts the start time (first number) to act as the split point,
    and checks the transient peak amplitude (last number). Drops points with amplitude < 0.001.
    Returns a sorted list of these partitions.
    """
    split_points = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # The last value in the row is the amplitude
                    amplitude = float(parts[-1])
                    if amplitude < 0.005:
                        continue
                        
                    start_sec = float(parts[0])
                    # Take the beginning of the transient band
                    split_points.append(start_sec)
                except ValueError:
                    continue
    return sorted(split_points)

def parse_vad_file(vad_path):
    """
    Parses the VAD lab file to extract silence intervals.
    Returns a list of tuples: (start_time, end_time) for 'SIL' segments.
    """
    silences = []
    with open(vad_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                    label = parts[2]
                    if label == "SIL":
                        silences.append((start_sec, end_sec))
                except ValueError:
                    continue
    return silences

def main():
    parser = argparse.ArgumentParser(description="Split WAV files, move silent split points to VOI poles, and drop short segments.")
    parser.add_argument('-i', '--input', required=True, help='Path to the input folder (must contain "audio", "seq/transient", and "seq/vad_loud").')
    parser.add_argument('-o', '--output', required=True, help='Path to the output folder.')
    parser.add_argument('-t', '--transient-version', default='W3_TH1.5_FFT4', 
                        help='Transient detection version folder name (default: "W3_TH1.5_FFT4").')
    parser.add_argument('-p', '--split-padding', default=0.0, 
                            help='Padding for segments(default=0.0)')
    

    args = parser.parse_args()

    input_folder = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)
    transient_version = args.transient_version

    audio_folder = os.path.join(input_folder, 'audio')
    lab_folder = os.path.join(input_folder, 'seq', 'transient', transient_version)
    vad_folder = os.path.join(input_folder, 'seq', 'vad_loud')
    lab_out_dir = os.path.join(input_folder, 'seq', 'lab_phon')

    os.makedirs(lab_out_dir, exist_ok=True)

    if not os.path.isdir(audio_folder):
        print(f"Error: Audio folder does not exist: {audio_folder}")
        return
    if not os.path.isdir(lab_folder):
        print(f"Error: Transient lab folder does not exist: {lab_folder}")
        return
    if not os.path.isdir(vad_folder):
        print(f"Error: VAD data folder does not exist at {vad_folder}. Please generate VAD data first.")
        return

    wav_files = [f for f in os.listdir(audio_folder) if f.lower().endswith('.wav')]
    if not wav_files:
        print(f"No .wav files found in {audio_folder}")
        return

    for wav_file in wav_files:
        original_name = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(audio_folder, wav_file)
        
        lab_candidates = [f for f in os.listdir(lab_folder) if original_name in f and f.lower().endswith('.lab')]
        vad_candidates = [f for f in os.listdir(vad_folder) if original_name in f and f.lower().endswith('.lab')]
        
        if not lab_candidates:
            print(f"Error: Mirror structure/lab file missing for '{wav_file}'. Expected in: {lab_folder}")
            continue
        if not vad_candidates:
            print(f"Error: VAD data missing for '{wav_file}'. Expected in: {vad_folder}")
            continue
            
        lab_path = os.path.join(lab_folder, lab_candidates[0])
        vad_path = os.path.join(vad_folder, vad_candidates[0])
            
        print(f"Processing '{wav_file}'...")
        
        partitions_sec = parse_lab_file(lab_path)
        
        if not partitions_sec:
            print(f"  Warning: No valid partitions found in {lab_path}. Skipping.")
            continue
        
        try:
            sample_rate, data = wavfile.read(wav_path)
        except Exception as e:
            print(f"  Error reading {wav_file}: {e}")
            continue

        total_samples = len(data)
        total_duration_sec = total_samples / sample_rate

        # Override first partition to 0.0 and append song end if not present
        if partitions_sec[0] != 0.0:
            partitions_sec[0] = 0.0
        if partitions_sec[-1] != total_duration_sec:
            partitions_sec.append(total_duration_sec)

        file_output_dir = os.path.join(output_folder, original_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        lab_out_filepath = os.path.join(lab_out_dir, f"{original_name}.seg.lab")

        chunks_saved = 0
        chunk_number = 1
        
        with open(lab_out_filepath, 'w') as lab_out_file:
            for i in range(len(partitions_sec) - 1):
                start_sec = partitions_sec[i]
                end_sec = partitions_sec[i+1]
                
                # Skip empty slices
                if start_sec >= end_sec:
                    continue
                    
                # Filter segments shorter than 30 ms BEFORE applying padding
                duration_sec = end_sec - start_sec
                if duration_sec < 0.03:
                    continue
                    
                chunk_filename = f"{original_name}_{chunk_number}.wav"
                
                # Write original, unpadded boundaries to the intermediate lab
                lab_out_file.write(f"{start_sec:.6f} {end_sec:.6f} {chunk_filename}\n")
                
                # Apply 10ms padding to the extraction boundaries
                pad = args.split_padding
                start_sec_padded = max(0.0, start_sec - pad)
                end_sec_padded = min(total_duration_sec, end_sec + pad)
                
                # Convert padded boundaries to sample indices
                start_sample = int(start_sec_padded * sample_rate)
                end_sample = int(end_sec_padded * sample_rate)
                
                chunk_data = data[start_sample:end_sample]
                chunk_filepath = os.path.join(file_output_dir, chunk_filename)
                
                wavfile.write(chunk_filepath, sample_rate, chunk_data)
                chunks_saved += 1
                chunk_number += 1
                
        print(f"  Successfully saved {chunks_saved} chunks to {file_output_dir}/ and lab to {lab_out_filepath}")

if __name__ == "__main__":
    main()