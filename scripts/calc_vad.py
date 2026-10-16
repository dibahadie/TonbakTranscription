import os
import argparse
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

def get_loudness_from_scratch(audio, sr, win_len_s=0.030, step_s=0.005, alpha=0.30103):
    """
    Computes an approximated loudness curve strictly using standard SciPy/NumPy.
    Applies Stevens' Power Law (alpha=0.30103) to the spectral intensity.
    """
    # Calculate window size and FFT parameters matching the original method
    win_len = int(2 * (win_len_s * sr / 2)) + 1
    nperseg = win_len
    noverlap = win_len - int(step_s * sr)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(audio, fs=sr, window='hann', 
                                          nperseg=nperseg, noverlap=noverlap)
    
    # Get power spectrum (intensity)
    power_spec = np.square(np.abs(Zxx))
    
    # Approximate specific loudness without external filterbanks by taking the 
    # intensity raised to the alpha power (Stevens' Power Law)
    # Adding a tiny epsilon prevents log/power errors on digital zero
    loudCurve = np.sum(np.power(power_spec + 1e-12, alpha), axis=0) * 1000 
    
    return times, loudCurve

def search_thresh_bisect(sequence, thresh_steps, std_red_limit):
    """
    Finds the silence threshold using bisectioning based on the reduction 
    in standard deviation of the loudness contour.
    """
    min_l_threshold = np.min(sequence)
    min_thresh_std = np.std(np.fmax(sequence, min_l_threshold))
    std_limit = min_thresh_std * std_red_limit
    max_l_threshold = np.max(sequence)

    for _ in range(thresh_steps):
        test_threshold = (min_l_threshold + max_l_threshold) / 2.0
        test_std = np.std(np.fmax(sequence, test_threshold))
        
        if test_std > std_limit:
            min_l_threshold = test_threshold
        elif test_std < std_limit:
            max_l_threshold = test_threshold
        else:
            min_l_threshold = max_l_threshold = test_threshold
            break

    return min_l_threshold

def vad_fuse_activity(hop_len_s, active_v, silence_duration_min_s, speech_duration_min_s):
    """
    Fuses active segments directly using NumPy arrays.
    """
    active_out_v = np.zeros_like(active_v)
    
    # Find start and end indices of activity
    dd = np.diff(np.concatenate(([0], active_v, [0])))
    voiced_starts = np.flatnonzero(dd > 0)
    voiced_ends = np.flatnonzero(dd < 0)
    
    voiced_starts_cand = []
    voiced_ends_cand = []
    
    # First pass: add voiced segments longer than requested
    for sv, ev in zip(voiced_starts, voiced_ends):
        if (ev - sv) * hop_len_s > speech_duration_min_s:
            sv_low_pass = int(max(0, sv - (silence_duration_min_s / hop_len_s)))
            
            # Look backwards for an existing active segment to connect to
            idx = np.where(active_out_v[sv_low_pass:sv] == 1)[0]
            if len(idx) > 0:
                ind = sv_low_pass + idx[-1]
                active_out_v[ind:ev] = 1
            else:
                active_out_v[sv:ev] = 1
        else:
            voiced_starts_cand.append(sv)
            voiced_ends_cand.append(ev)
            
    # Second pass: integrate active segments closer to an already selected segment
    done = False
    while not done:
        done = True
        voiced_starts_unused = []
        voiced_ends_unused = []
        
        for sv, ev in zip(voiced_starts_cand, voiced_ends_cand):
            sv_low = int(max(0, sv - (silence_duration_min_s / hop_len_s)))
            sv_high = int(min(len(active_out_v), ev + (silence_duration_min_s / hop_len_s)))
            used = False
            
            # Check backwards
            idx_low = np.where(active_out_v[sv_low:sv] == 1)[0]
            if len(idx_low) > 0:
                active_out_v[sv_low + idx_low[-1]:ev] = 1
                used = True
                done = False
                
            # Check forwards
            if not used:
                idx_high = np.where(active_out_v[ev:sv_high] == 1)[0]
                if len(idx_high) > 0:
                    active_out_v[sv:ev + idx_high[0]] = 1
                    used = True
                    done = False
                    
            if not used:
                voiced_starts_unused.append(sv)
                voiced_ends_unused.append(ev)
                
        voiced_starts_cand = voiced_starts_unused
        voiced_ends_cand = voiced_ends_unused
        
    return active_out_v

def get_active_segs_from_activity(ts_v, active_v, last_samp_s):
    """
    Converts the binary activity array into timestamps labeled as VOI or SIL.
    """
    active_segs = []
    is_active = bool(active_v[0])
    seg_start_s = ts_v[0]

    for ltime, act in zip(ts_v, active_v):
        if is_active:
            if not act:
                is_active = False
                active_segs.append((seg_start_s, ltime, "VOI"))
                seg_start_s = ltime
        else:
            if act:
                is_active = True
                active_segs.append((seg_start_s, ltime, "SIL"))
                seg_start_s = ltime

    seg_end_s = ts_v[-1] + (ts_v[-1] - ts_v[-2]) if len(ts_v) > 1 else ts_v[0]
    seg_end_s = min(seg_end_s, last_samp_s)
        
    if is_active:
        active_segs.append((seg_start_s, seg_end_s, "VOI"))
    else:
        active_segs.append((seg_start_s, seg_end_s, "SIL"))

    return active_segs

def main():
    parser = argparse.ArgumentParser(description="Batch process audio for VAD using the loud method logic from scratch.")
    parser.add_argument('-i', '--input', required=True, help="Path to input folder containing the 'audio' subfolder.")
    
    # Original algorithm thresholds mapped for this use case
    parser.add_argument("--loud_alpha", default=0.30103, type=float)
    parser.add_argument("--loud_silence_sone_modulation_fac", default=2.0, type=float)
    parser.add_argument("--silence_thresh_auto_std", default=0.95, type=float)
    parser.add_argument("--n_grid_steps", default=1000, type=int)
    parser.add_argument("--win_len_s", default=0.030, type=float)
    parser.add_argument("--step_s", default=0.005, type=float)
    parser.add_argument("--speech_duration_min_s", default=0.050, type=float)
    parser.add_argument("--silence_duration_min_s", default=0.150, type=float)

    args = parser.parse_args()
    
    input_folder = os.path.abspath(args.input)
    audio_dir = os.path.join(input_folder, 'audio')
    out_dir = os.path.join(input_folder, 'seq', 'vad_loud')
    
    if not os.path.isdir(audio_dir):
        print(f"Error: Audio folder does not exist at {audio_dir}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]

    if not audio_files:
        print(f"No .wav files found in {audio_dir}")
        return

    for audio_file in audio_files:
        infile = os.path.join(audio_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        outfile = os.path.join(out_dir, f"{base_name}.vad.lab")
        
        print(f"Processing '{audio_file}'...")
        
        try:
            # Read standard .wav
            sr, audio = wav.read(infile)
            
            # Convert to float and mono if needed
            audio = audio.astype(np.float32)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
                
            last_samp_s = len(audio) / sr
            
            # Step 1: Calculate loudness from scratch
            time_v, loud_v = get_loudness_from_scratch(
                audio, sr, 
                win_len_s=args.win_len_s, 
                step_s=args.step_s, 
                alpha=args.loud_alpha
            )
            
            # Step 2: Bisection to find optimal silence threshold
            loud_silence_thresh_sone = search_thresh_bisect(
                loud_v,
                int(np.ceil(np.log2(args.n_grid_steps))) + 1,
                args.silence_thresh_auto_std
            )

            # Step 3: Modulated threshold state machine
            modulation_thresh = loud_silence_thresh_sone * args.loud_silence_sone_modulation_fac
            is_silence = loud_v[0] <= modulation_thresh
            active_v = np.zeros(len(time_v), dtype=np.int32)
            
            for ii, lsone in enumerate(loud_v):
                if is_silence:
                    if lsone > modulation_thresh:
                        is_silence = 0
                else:
                    if lsone < loud_silence_thresh_sone:
                        is_silence = 1
                active_v[ii] = not is_silence

            # Step 4: Minimum duration fusion
            if args.speech_duration_min_s > 0 or args.silence_duration_min_s > 0:
                active_v = vad_fuse_activity(
                    hop_len_s=args.step_s, 
                    active_v=active_v, 
                    speech_duration_min_s=args.speech_duration_min_s,
                    silence_duration_min_s=args.silence_duration_min_s
                )

            # Step 5: Format into segments and save
            active_segs = get_active_segs_from_activity(time_v, active_v, last_samp_s)

            with open(outfile, 'w') as f:
                for start, end, label in active_segs:
                    f.write(f"{start:.6f} {end:.6f} {label}\n")
                    
            print(f"  -> Saved to {outfile}")
            
        except Exception as e:
            print(f"  -> Error processing {audio_file}: {e}")

if __name__ == "__main__":
    main()