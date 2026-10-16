import os
import glob
import argparse
import sys

def read_lab_file(filepath):
    intervals = []
    if not os.path.exists(filepath):
        return None 
        
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = " ".join(parts[2:])
                    intervals.append((start, end, label))
                except ValueError:
                    continue
    return intervals

def apply_roll_overrides(transients, rolls):
    # Only care about actual 'roll' segments
    actual_rolls = [r for r in rolls if r[2].lower() == 'roll']
    
    # Rename ALL incoming transient labels to 'seg' right away
    current_transients = [(t_start, t_end, 'seg') for t_start, t_end, _ in transients]
    
    for r_start, r_end, r_label in actual_rolls:
        next_transients = []
        
        for t_start, t_end, t_label in current_transients:
            # 1. Segment completely outside the roll -> Keep it completely
            if t_end <= r_start or t_start >= r_end:
                next_transients.append((t_start, t_end, t_label))
                
            # 2. Segment completely falls inside the roll -> Delete it
            elif t_start >= r_start and t_end <= r_end:
                continue 
                
            # 3. Left overlap -> Keep part outside (t_start to r_start)
            elif t_start < r_start and t_end <= r_end:
                next_transients.append((t_start, r_start, t_label))
                
            # 4. Right overlap -> Keep part outside (r_end to t_end)
            elif t_start >= r_start and t_end > r_end:
                next_transients.append((r_end, t_end, t_label))
                
            # 5. Transient completely surrounds the roll -> Keep both parts outside
            elif t_start < r_start and t_end > r_end:
                next_transients.append((t_start, r_start, t_label))
                next_transients.append((r_end, t_end, t_label))
                
        current_transients = next_transients
        
    # Combine surviving original segments with the roll segments
    final_intervals = current_transients + actual_rolls
    
    # Sort chronologically by start time
    final_intervals.sort(key=lambda x: x[0])
    
    return final_intervals

def remove_short_segments(intervals, min_duration=0.020):
    if not intervals:
        return []
        
    segments = [list(iv) for iv in intervals]
    
    while True:
        found_short = False
        
        for i in range(len(segments)):
            start, end, label = segments[i]
            
            if end - start < min_duration:
                found_short = True
                
                # Cannot merge if it's the only segment
                if len(segments) == 1:
                    found_short = False
                    break
                    
                mid = (start + end) / 2.0
                
                if i == 0:
                    segments[i+1][0] = start
                elif i == len(segments) - 1:
                    segments[i-1][1] = end
                else:
                    segments[i-1][1] = mid
                    segments[i+1][0] = mid
                    
                segments.pop(i)
                break 
                
        if not found_short:
            break
            
    return [(s[0], s[1], s[2]) for s in segments]

def shift_starts_earlier(intervals, shift_amount=0.010):
    if not intervals:
        return []
        
    segments = [list(iv) for iv in intervals]
    
    for i in range(len(segments)):
        # Only shift the starts of the original segments (now labeled 'seg')
        if segments[i][2] == 'seg':
            original_start = segments[i][0]
            
            # Prevent shifting into negative time
            new_start = max(0.0, original_start - shift_amount)
            
            segments[i][0] = new_start
            
            # If there's a preceding segment, pull its end back so they stay contiguous
            if i > 0:
                # Only pull it back if they were originally touching (to be safe)
                if abs(segments[i-1][1] - original_start) < 1e-6:
                    segments[i-1][1] = new_start

    # Final pass to ensure no segment was inverted or zeroed out
    valid_segments = []
    for s in segments:
        if s[1] > s[0]:
            valid_segments.append((s[0], s[1], s[2]))
            
    return valid_segments

def process_files(audio_folder, transient_folder, roll_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    
    if not wav_files:
        print(f"CRITICAL ERROR: No .wav files found in {audio_folder}")
        sys.exit(1)
        
    for wav_filepath in wav_files:
        file_name = os.path.splitext(os.path.basename(wav_filepath))[0]
        
        # --- FIND TRANSIENT FILE ---
        transient_path = os.path.join(transient_folder, f"{file_name}.seg.lab")
        transients = read_lab_file(transient_path)
        
        if transients is None:
            print(f"ERROR: Could not find transient file: {transient_path}. Skipping this file.")
            continue
        if len(transients) == 0:
            print(f"WARNING: Transient file {transient_path} is completely empty.")
            
        # --- FIND ROLL FILE ---
        roll_path = os.path.join(roll_folder, f"{file_name}_converted.lab")
        if not os.path.exists(roll_path):
            roll_path = os.path.join(roll_folder, f"{file_name}.lab")
            
        rolls = read_lab_file(roll_path)
        
        if rolls is None:
            print(f"ERROR: Could not find roll file for {file_name}. Skipping.")
            continue
            
        # 1. Overlay rolls on transients & rename transients to 'seg'
        merged = apply_roll_overrides(transients, rolls)
        
        # 2. Cleanup anything under 20ms
        cleaned = remove_short_segments(merged, min_duration=0.020)
        
        # 3. Shift 'seg' starts 10ms earlier
        final_timeline = shift_starts_earlier(cleaned, shift_amount=0.010)
        
        out_filepath = os.path.join(output_folder, f"{file_name}.lab")
        with open(out_filepath, 'w') as f:
            for start, end, label in final_timeline:
                f.write(f"{start:.6f} {end:.6f} {label}\n")
                
        print(f"SUCCESS: {file_name} -> {out_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, help="Path to the folder containing .wav files")
    parser.add_argument("-t", "--transient", required=True, help="Path to the folder containing transient .seg.lab files")
    parser.add_argument("-r", "--roll", required=True, help="Path to the folder containing roll .lab files")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")
    
    args = parser.parse_args()
    process_files(args.audio, args.transient, args.roll, args.output)