# Face Identification & Clustering System

A computer vision system that automatically groups unorganized photos of multiple people into per-person clusters — robust to different lighting conditions, camera angles, and facial expressions — and assigns every image a confidence score reflecting how certain the system is about that identity match.

## Submission Notes

> **Important:** If the embedded or previous video link is inaccessible, please use the public Google Drive link below.

- **GitHub Repository:**  https://github.com/Harshvish26/FaceIdentificationClusteringSystem
- **Demo Video:** https://drive.google.com/file/d/1oXUe6YSPCUYiv9FXkMOz09Kv16WhOozw/view?usp=drive_link


Built for the **Computer Vision Engineer — Round 1 Assessment** at [Future of Gaming](https://www.futureofgaming.tech).

---

## Table of Contents

- [Overview](#overview)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Understanding the Output](#understanding-the-output)
- [Choosing the Right `--eps` Value](#choosing-the-right---eps-value)
- [Confidence Score — What It Means](#confidence-score--what-it-means)
- [Example Output](#example-output)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## Overview

Given a folder of unorganized images — where the same person may appear multiple times under different lighting, angles, and expressions, mixed in with photos of other people — this tool:

1. **Detects every face** in every image.
2. **Generates a numerical "fingerprint" (embedding)** for each face that captures identity-relevant features.
3. **Automatically clusters** these fingerprints so that all photos of the same person end up in one group — without needing to know in advance how many different people are in the dataset.
4. **Scores each photo** with a confidence percentage showing how strongly it matches the identity of its assigned group.
5. **Outputs** organized folders per person, a CSV report, and an interactive HTML visual report.

---

## How It Works (Pipeline)

```
 Input Images
      │
      ▼
┌─────────────────────┐
│  1. CLAHE Lighting   │   Corrects dim / colored / uneven lighting (e.g. club lighting)
│     Normalization    │   before detection, so poor lighting doesn't hurt accuracy.
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  2. Face Detection   │   MTCNN (facenet-pytorch) locates every face in the image
│     (MTCNN)          │   and returns a bounding box + detection confidence per face.
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  3. Face Embedding   │   InceptionResnetV1, pretrained on VGGFace2, converts each
│  (InceptionResnetV1) │   detected face into a 512-dimensional identity vector.
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  4. Clustering       │   DBSCAN groups embeddings by cosine distance. No need to
│     (DBSCAN)         │   specify the number of people in advance — it's detected
│                       │   automatically from the data.
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  5. Confidence Score │   Each face's cosine similarity to its cluster's centroid
│                       │   is converted into a 0–100% confidence score.
└─────────┬────────────┘
          ▼
  Organized Output Folders + CSV + HTML Report
```

**Why this approach is robust:**
- **Lighting variation** → handled by CLAHE normalization before detection.
- **Angle/expression variation** → handled by using a deep embedding model (VGGFace2-trained) rather than raw pixel comparison, since embeddings capture identity, not pose.
- **Group photos / bystanders** → by default, only the largest (main subject) face per image is kept, filtering out background people. This can be disabled with `--keep_all_faces`.
- **Unknown number of people** → DBSCAN discovers the number of clusters automatically; no need to pre-specify "there are N people."

---

## Project Structure

```
FaceIdentificationClusteringSystem/
│
├── face_cluster.py         # Main script — the entire pipeline
├── Requirements.txt         # Python dependencies
├── README.md                 # This file
│
├── dataset/                   # INPUT: place your unorganized images here
│   ├── person_01_0.jpg
│   ├── person_01_1.jpg
│   └── ...
│
└── clustered_output/         # OUTPUT: generated after running the script
    ├── person_0/               # All images identified as Individual #1
    ├── person_1/               # All images identified as Individual #2
    ├── unclustered/            # Faces that didn't confidently match any group
    ├── results.csv              # image, face_index, cluster_id, person_label, confidence_score
    └── report.html               # Visual, browsable report of all clusters
```

---

## Installation

**Requirements:** Python 3.9+ (tested on Windows with a virtual environment)

```powershell
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r Requirements.txt
```

If a package fails to install due to a network timeout, retry with:
```powershell
pip install -r Requirements.txt --timeout 120
```

---

## Usage

### 1. Add your images
Place all images (any mix of people, lighting, angles) into the `dataset/` folder. Subfolders are supported.

### 2. (Recommended) Check the natural separation in your data first
Before clustering, see how similar/different faces actually are — this tells you what `--eps` value to use:
```powershell
python face_cluster.py --input_dir dataset --debug_distances
```
This prints a full pairwise cosine-distance table between every detected face and exits (no clustering is performed).

### 3. Run the clustering
```powershell
python face_cluster.py --input_dir dataset --output_dir clustered_output --eps 0.45 --min_samples 2 --report
```
- `--report` automatically opens an HTML visual report in your browser when done.
- Omit `--eps` and add `--auto_eps` instead if you want the script to try to pick a value automatically (see [caveats](#choosing-the-right---eps-value) below).

### 4. Re-running
Always clear the previous output first, since cluster IDs (`person_0`, `person_1`, ...) can shift between runs:
```powershell
Remove-Item -Recurse -Force clustered_output
python face_cluster.py --input_dir dataset --output_dir clustered_output --eps 0.45 --min_samples 2 --report
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | *required* | Folder containing unorganized input images |
| `--output_dir` | `clustered_output` | Where clustered results are written |
| `--eps` | `0.45` | DBSCAN cosine-distance threshold. Lower = stricter matching (fewer false merges, more false splits) |
| `--min_samples` | `2` | Minimum images needed to form a cluster |
| `--auto_eps` | off | Automatically sweep a range of `--eps` values and pick the one with the best silhouette score, instead of a fixed `--eps`. *(See caveats below — manual `--eps` via `--debug_distances` is generally more reliable on small datasets.)* |
| `--debug_distances` | off | Print the full pairwise cosine-distance matrix and exit, without clustering. Use this to pick `--eps`. |
| `--keep_all_faces` | off | Keep every detected face per image instead of only the largest (main subject). Useful if you want to cluster bystanders/background people too. |
| `--margin` | `40` | Extra pixel context around each detected face before embedding |
| `--image_size` | `160` | Size faces are resized to before embedding (FaceNet's native input size) |
| `--embedding_model` | `vggface2` | Pretrained weights: `vggface2` or `casia-webface` |
| `--no_enhance` | off | Disable CLAHE lighting normalization (enabled by default) |
| `--device` | auto | Force `cuda` or `cpu` (auto-detected if not set) |
| `--report` | off | Generate `report.html` and open it automatically |

---

## Understanding the Output

**`clustered_output/results.csv`**

| Column | Meaning |
|---|---|
| `image` | Original filename |
| `face_index` | Index of the face within that image (0 if only one face kept) |
| `cluster_id` | Numeric cluster ID (`-1` = unclustered) |
| `person_label` | `person_0`, `person_1`, ... or `unclustered` |
| `confidence_score` | 0–100%, how strongly this face matches its assigned person |

**`clustered_output/person_N/`** — copies of every image identified as that individual.

**`clustered_output/unclustered/`** — faces that didn't confidently match any group (e.g. only one photo of that person exists, so no pair could form a cluster given `--min_samples`).

**`clustered_output/report.html`** — a dark-themed visual dashboard: overall stats (total faces, individuals found, average confidence) at the top, followed by one section per person showing every photo with its confidence score and a color-coded progress bar (green = high, yellow = medium, red = low confidence).

---

## Choosing the Right `--eps` Value

`--eps` is the cosine-distance threshold DBSCAN uses to decide "these two faces are the same person." Getting it right matters more than any other setting.

**Recommended workflow:**
1. Run `python face_cluster.py --input_dir dataset --debug_distances`.
2. Look at the printed distance table:
   - Find the **largest distance between two photos you know are the same person** (same-person pairs).
   - Find the **smallest distance between two photos of different people** (different-person pairs).
3. Pick an `--eps` value **between these two numbers** — ideally closer to the middle of that gap.

**Example:** if same-person pairs never exceed `0.34` and different-person pairs are never below `0.67`, any `--eps` between `0.34` and `0.67` (e.g. `0.45`, the default) will cluster correctly.

**Why not just always use `--auto_eps`?** It picks the value that gives the best *overall* silhouette score across the whole dataset, which can occasionally be too strict for a specific pair (e.g. a blurry/angled photo) even though a manually-chosen value works fine for everyone. On small or tricky datasets, checking `--debug_distances` and setting `--eps` manually is the more reliable choice.

---

## Confidence Score — What It Means

The confidence score is **not** the same as face-detection confidence (i.e. "is there a face here"). It measures **identity-match confidence**:

> *Cosine similarity between this face's embedding and the average embedding (centroid) of all faces in its assigned cluster, expressed as a percentage.*

- **High (≥70%)** — strongly matches its group; a confident identity match.
- **Medium (40–69%)** — plausible match, but with more visual variation (angle, lighting, expression) than a typical same-person pair.
- **Low (<40%)** — weak match. For unclustered faces, this instead shows similarity to the closest other face found in the whole dataset, since an unclustered face has no group centroid to compare against.

---

## Example Output

Running on a small 3-person test set:

```
===== SUMMARY =====
Total faces processed : 6
Individuals found      : 3
Unclustered faces      : 0
  person_0       :   2 images | avg confidence 92.7%
  person_1       :   2 images | avg confidence 91.2%
  person_2       :   2 images | avg confidence 95.8%
====================
```

Every individual correctly grouped, with confidently high match scores.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| A person's images are split into two clusters | `--eps` too strict for that pair | Run `--debug_distances`, check the true same-person distance, raise `--eps` slightly |
| Two different people merged into one cluster | `--eps` too loose | Lower `--eps` based on `--debug_distances` output |
| A person with only one usable photo shows up as "unclustered" | `--min_samples 2` requires at least 2 photos to form a cluster | Expected behavior — DBSCAN can't form a group from a single point. Lower `--min_samples` to `1` only if you want singletons treated as their own cluster |
| `pip install` fails with a `ReadTimeoutError` | Slow/unstable internet mid-download | Re-run `pip install -r Requirements.txt --timeout 120`; already-downloaded packages are cached and skipped |
| A genuine face gets skipped entirely | (Fixed) — earlier versions filtered out faces below a fixed MTCNN detection-confidence threshold. The current version keeps every detected face; low-quality detections simply end up with a lower confidence score instead of being dropped | N/A — already handled in this version |

---

## Tech Stack

- **Python 3**
- **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** — MTCNN (face detection) + InceptionResnetV1 (face embeddings, VGGFace2-pretrained)
- **PyTorch / Torchvision** — model backend
- **OpenCV** — CLAHE lighting normalization
- **scikit-learn** — DBSCAN clustering, cosine distance/similarity, silhouette scoring
- **NumPy, Pillow** — array and image handling

---
