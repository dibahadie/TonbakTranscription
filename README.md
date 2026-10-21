# Tonbak Rhythm Transcription & Stroke Clustering Pipeline

An automated and semi-automated audio processing pipeline designed to segment, cluster, and transcribe traditional Persian Tonbak percussion recordings. 

Tonbak strokes can be highly ambiguous and difficult to transcribe manually (e.g., differentiating between a high-frequency *Bak* and *Beshkan*). This toolkit addresses that by combining manual annotations with unsupervised deep-learning clustering. It extracts musical representation using the **MERT**, clusters similar strokes using GMM, and outputs clean, symbolic rhythm sequences ready for downstream machine learning or musical analysis.

---

## Data Origins & Preparation

The pipeline is designed to fuse automated transient detection with manual human annotations to create a highly accurate ground-truth timeline.

*   **Transient Detection (Automated):** Initial percussion hits and transients are detected using **partials**. The outputs of these transient detections are saved in the `seq/transient/<tag>` directory, where `<tag>` represents the specific configuration settings (e.g., threshold, FFT size) used during extraction.
*   **Roll Annotations (Manual):** Rolls (trills/riz) are notoriously difficult to capture via standard transient detection. These are transcribed manually using **dfasma**. In their raw form, these annotations use a simple `s` (start) and `e` (end) structure to mark the boundaries of a roll section.
*   **Label Format:** Throughout the pipeline, data is stored in Audacity-style three-column label files (`.lab`), formatted as: `start_time end_time label`.

---

## Pipeline Workflow & Script Details

The pipeline is broken down into four main phases: Preparation, Merging, Clustering, and Export. You can run the entire workflow automatically using the provided `process_dataset.sh` bash script, or run the Python scripts individually as detailed below.

### Phase 1: Data Preprocessing & VAD

**1. Voice Activity Detection (`calc_vad.py`)**
```bash
python scripts/calc_vad.py -i ./data/dataset_folder 
```

**2. Roll Marker Conversion (`convert_manual_roll_labels.py`)**
Converts the raw `s` and `e` dfasma roll markers into continuous, standard `.lab` intervals (e.g., `roll` and `not_roll`).
```bash
python scripts/convert_manual_roll_labels.py ./data/dataset_folder/seq/lab_roll/manual
```

### Phase 2: Merging Timelines

**3. Transient & Roll Fusion (`merge_transient_roll.py`)**
Takes the IRCAM transient detections and overlays the manual roll annotations. Roll annotations take priority; any transients overlapping a roll are trimmed or deleted. It also cleans up the timeline by removing short segments (<20ms) and shifting transient starts slightly earlier to ensure the full attack is captured.
```bash
python scripts/merge_transient_roll.py \
  -a ./data/dataset_folder/audio \
  -t ./data/dataset_folder/seq/transient/W3_TH1.5_FFT4 \
  -r ./data/dataset_folder/seq/lab_roll/manual \
  -o ./data/dataset_folder/seq/lab_seg
```

### Phase 3: Clustering (MERT)

This phase feeds the separated percussion segments into **MERT-v1-95M** to extract 768-dimensional acoustic embeddings. The pipeline offers two clustering scripts depending on your dataset size:

**Option A: PCA-Reduced Clustering (`cluster_MERT_PCA.py`) - *Recommended***
Best for small to medium datasets. It applies Principal Component Analysis (PCA) to reduce the 768 MERT dimensions down to ≤32 components before fitting a diagonal Gaussian Mixture Model. This prevents mathematical instability and "singular covariance matrix" errors.
```bash
python scripts/cluster_MERT_PCA.py \
  --audio_dir ./data/dataset_folder/audio \
  --label_dir ./data/dataset_folder/seq/lab_seg \
  --output_dir ./data/dataset_folder/seq/lab_phon \
  --n_clusters 2
```

**Option B: Standard Clustering (`cluster_MERT.py`)**
Uses the raw 768-dimensional space with a full-covariance GMM. Only recommended if you have thousands of transient segments, allowing the model to safely estimate the massive covariance matrix.
```bash
python scripts/cluster_MERT.py \
  --audio_dir ./data/dataset_folder/audio \
  --label_dir ./data/dataset_folder/seq/lab_seg \
  --output_dir ./data/dataset_folder/seq/lab_phon \
  --n_clusters 3
```
*Both scripts output renamed segments (e.g., `cluster_0`, `cluster_1`) and a 2D t-SNE scatter plot (`global_clusters.png`) to visualize cluster separation.*

### Phase 4: Label Refinement & Export

**4. Simplify Stroke Taxonomy (`replace_tom.py`)**
Numerical clusters aren't useful for transcription. By referencing the t-SNE plot and listening to audio samples, identify the cluster representing the low/center bass hit. This script maps that specific cluster ID to `tom`, and defaults the remaining perimeter clusters (like *Bak*/*Beshkan*) to `edge`.
```bash
python scripts/replace_tom.py \
  --input_dir ./data/dataset_folder/seq/lab_phon \
  --tom_cluster 1 \
  --other_label edge
```

**5. Roll Section Separation (`create_roll_sections.py`)**
If rolls were clustered rather than manually merged, this script can parse the clustered `.lab` files and extract contiguous blocks of specific clusters into broader `roll` and `hit` sections.
```bash
python scripts/create_roll_sections.py \
  -i ./data/dataset_folder \
  -c 2
```

**6. Sequence Extraction (`extract_transcription.py`)**
Strips all start/end timestamps from the final `.partial.lab` files, exporting clean text files containing only the chronological stroke sequence (e.g., `tom edge edge tom roll`).
```bash
python scripts/extract_transcription.py \
  --input_dir ./data/dataset_folder/seq/lab_phon \
  --output_dir ./data/dataset_folder/txt \
  --separator " "
```

---

## Full Automation

You can run the entire workflow seamlessly using the main bash wrapper. Edit the variables at the top of the script to point to your data directories, then execute:

```bash
bash scripts/process_dataset.sh