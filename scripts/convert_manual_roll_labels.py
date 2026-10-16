import os
import glob
import argparse
import wave

def get_wav_duration(wav_filepath):
    """Returns the duration of a wav file in seconds, or None if it fails."""
    try:
        with wave.open(wav_filepath, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"  -> Warning: Could not read audio duration for {os.path.basename(wav_filepath)}: {e}")
        return None

def convert_lab_files(directory, force):
    # Resolve the path to the audio directory
    audio_dir = os.path.normpath(os.path.join(directory, "../../../audio"))
    
    if not os.path.exists(audio_dir):
        print(f"Error: Could not find audio directory at {audio_dir}")
        return
        
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        return
        
    for wav_filepath in wav_files:
        file_name = os.path.splitext(os.path.basename(wav_filepath))[0]
        
        lab_filepath = os.path.join(directory, file_name + ".lab")
        out_filepath = os.path.join(directory, file_name + "_converted.lab")
        
        # Check for force flag if output already exists
        if os.path.exists(out_filepath) and not force:
            print(f"Skipping {file_name}: Output already exists. Use --force to overwrite.")
            continue
            
        total_duration = get_wav_duration(wav_filepath)
        if total_duration is None:
            continue
            
        # 1. Parse the timestamps and labels if the original lab file exists
        events = []
        if os.path.exists(lab_filepath):
            with open(lab_filepath, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        time_val = float(parts[0])
                        label = parts[1]
                        events.append((time_val, label))
                    except ValueError:
                        continue
                        
        # 2. Build the continuous timeline
        intervals = []
        
        if not events:
            # If no lab file exists or it's empty, the whole file is not_roll
            intervals.append((0.0, total_duration, "not_roll"))
        else:
            current_time = 0.0
            i = 0
            while i < len(events):
                time_val, label = events[i]
                
                if label == 's':
                    # Gap between current time and this start is 'not_roll'
                    if time_val > current_time:
                        intervals.append((current_time, time_val, "not_roll"))
                    
                    # The next event should be the corresponding 'e' (end)
                    if i + 1 < len(events) and events[i+1][1] == 'e':
                        e_time = events[i+1][0]
                        intervals.append((time_val, e_time, "roll"))
                        current_time = e_time
                        i += 2 # Skip the 'e' since we just processed it
                    else:
                        print(f"  -> Warning: Missing 'e' after 's' at {time_val} in {lab_filepath}")
                        break
                else:
                    # Stray 'e' or malformed data
                    i += 1
                    
            # 3. Fill the end of the file with not_roll if needed
            if current_time < total_duration:
                intervals.append((current_time, total_duration, "not_roll"))
                
        # 4. Save the new format
        if intervals:
            with open(out_filepath, 'w') as f:
                for start, end, lbl in intervals:
                    f.write(f"{start:.6f} {end:.6f} {lbl}\n")
                    
            print(f"Converted: {file_name}.wav -> {os.path.basename(out_filepath)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert roll lab files to continuous start-end intervals and pad missing files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing the .lab files")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite of existing _converted.lab files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        convert_lab_files(args.directory, args.force)
        print("Conversion complete!")
    else:
        print(f"Error: '{args.directory}' is not a valid directory.")