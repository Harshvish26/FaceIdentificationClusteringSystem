import argparse
import csv
import shutil
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_lighting(pil_img: Image.Image) -> Image.Image:
    """
    Applies CLAHE (contrast-limited adaptive histogram equalization) on the
    luminance channel to correct for dim, colored, or unevenly lit photos
    (e.g. club/party lighting) before face detection & embedding. This
    reduces color-cast/brightness noise that would otherwise hurt embedding
    quality on low-light shots.
    """
    arr = np.array(pil_img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(arr)


def parse_args():
    p = argparse.ArgumentParser(description="Face identification & clustering")
    p.add_argument("--input_dir", required=True, help="Folder of unorganized images")
    p.add_argument("--output_dir", default="clustered_output", help="Where to write clustered results")
    p.add_argument("--eps", type=float, default=0.45,
                   help="DBSCAN cosine-distance threshold. Lower = stricter matching (default 0.45)")
    p.add_argument("--min_samples", type=int, default=2,
                   help="Minimum images to form a cluster (default 2)")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detected if not set)")
    p.add_argument("--keep_all_faces", action="store_true",
                   help="Keep every detected face per image (default: keep only the largest face, "
                        "i.e. the main subject closest to the camera — filters out bystanders in the background)")
    p.add_argument("--debug_distances", action="store_true",
                   help="Print the full pairwise cosine-distance matrix between all detected faces, "
                        "then exit before clustering. Use this to pick the exact --eps value.")
    p.add_argument("--margin", type=int, default=40,
                   help="Extra pixels of context around each detected face before embedding (default 40). "
                        "More context can help the embedding model on blurry/extreme-expression photos.")
    p.add_argument("--image_size", type=int, default=160,
                   help="Size faces are resized to before embedding (default 160, FaceNet's native size).")
    p.add_argument("--embedding_model", default="vggface2", choices=["vggface2", "casia-webface"],
                   help="Pretrained weights for the embedding model (default vggface2).")
    p.add_argument("--no_enhance", action="store_true",
                   help="Disable CLAHE lighting normalization (enabled by default). "
                        "Normalization helps on dim/colored-lighting photos.")
    p.add_argument("--auto_eps", action="store_true",
                   help="Automatically choose the best --eps by sweeping a range of values and "
                        "picking the one with the highest silhouette score, instead of using a fixed --eps.")
    p.add_argument("--report", action="store_true",
                   help="Generate an HTML visual report (report.html) of the clustered results and open it.")
    return p.parse_args()


def collect_images(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def extract_faces_and_embeddings(image_paths, device, keep_all_faces=False,
                                  margin=40, image_size=160, embedding_model="vggface2",
                                  enhance=True):
    """
    Runs face detection + embedding extraction over every image.
    An image can contain more than one face; each detected face becomes
    its own record so group photos are handled correctly.
    """
    mtcnn = MTCNN(keep_all=True, device=device, post_process=True,
                  margin=margin, image_size=image_size)
    resnet = InceptionResnetV1(pretrained=embedding_model).eval().to(device)

    records = []  # each: {"path": Path, "face_idx": int, "embedding": np.array, "box": [...]}

    for idx, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {path.name}: {e}")
            continue

        if enhance:
            img = normalize_lighting(img)

        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            print(f"[INFO] No face found in {path.name}, skipping.")
            continue

        faces = mtcnn.extract(img, boxes, save_path=None)
        if faces is None:
            continue

        with torch.no_grad():
            embeddings = resnet(faces.to(device)).cpu().numpy()

        # No confidence-based skipping: every face MTCNN detects (that isn't None)
        # is kept. A genuinely low-quality face will naturally end up with a low
        # confidence_score at the clustering stage instead of being silently
        # dropped here.
        candidates = []
        for face_idx, (emb, box, prob) in enumerate(zip(embeddings, boxes, probs)):
            if prob is None:
                continue  # MTCNN found no face at all in this box slot
            candidates.append({
                "path": path,
                "face_idx": face_idx,
                "embedding": emb,
                "box": box,
                "detection_prob": float(prob),
            })

        if not candidates:
            print(f"[INFO] No confident face found in {path.name}, skipping.")
            continue

        if not keep_all_faces:
            # Keep only the largest face = main subject closest to the camera.
            # Filters out bystanders/background people in group settings.
            candidates = [max(candidates, key=lambda c: box_area(c["box"]))]
            candidates[0]["face_idx"] = 0

        records.extend(candidates)
        print(f"[{idx}/{len(image_paths)}] {path.name}: {len(candidates)} face(s) kept "
              f"(of {len(embeddings)} detected)")

    return records


def auto_select_eps(dist_matrix, min_samples, eps_range=None):
    """
    Sweeps a range of DBSCAN eps values and picks the one that maximizes the
    silhouette score (a measure of how well-separated the resulting clusters
    are), instead of relying on a single manually guessed threshold.
    Falls back to a sensible default if no valid multi-cluster solution is found
    (e.g. too few images to form more than one cluster).
    """
    if eps_range is None:
        eps_range = np.arange(0.20, 0.90, 0.02)

    best_eps, best_score, best_labels = None, -1.0, None

    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(dist_matrix)
        mask = labels != -1
        unique_labels = set(labels[mask])
        if len(unique_labels) < 2 or mask.sum() < 2:
            continue  # silhouette needs at least 2 clusters with assigned points
        try:
            score = silhouette_score(dist_matrix[mask][:, mask], labels[mask], metric="precomputed")
        except ValueError:
            continue
        if score > best_score:
            best_eps, best_score, best_labels = float(eps), score, labels

    if best_eps is None:
        print("[WARN] auto_eps: no multi-cluster solution found in sweep range; falling back to eps=0.45")
        return 0.45

    print(f"[INFO] auto_eps selected eps={best_eps:.2f} (silhouette score={best_score:.3f})")
    return best_eps


def cluster_faces(records, eps, min_samples):
    embeddings = np.stack([r["embedding"] for r in records])
    # L2-normalize so cosine distance behaves well
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dist_matrix = cosine_distances(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(dist_matrix)

    for record, label in zip(records, labels):
        record["cluster_id"] = int(label)

    return records, embeddings


def compute_confidence_scores(records, embeddings):
    """Confidence = cosine similarity of a face's embedding to its cluster's centroid,
    expressed as a 0-100 percentage. Unclustered (-1) faces get a similarity to the
    nearest other face instead, since they have no centroid."""
    cluster_ids = sorted(set(r["cluster_id"] for r in records if r["cluster_id"] != -1))
    centroids = {}
    for cid in cluster_ids:
        idxs = [i for i, r in enumerate(records) if r["cluster_id"] == cid]
        centroid = embeddings[idxs].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cid] = centroid

    for i, r in enumerate(records):
        if r["cluster_id"] == -1:
            # No cluster: report similarity to the closest other face found overall
            sims = cosine_similarity(embeddings[i:i + 1], embeddings)[0]
            sims[i] = -1  # exclude self
            r["confidence"] = round(float(max(sims.max(), 0.0)) * 100, 2)
        else:
            centroid = centroids[r["cluster_id"]]
            sim = float(cosine_similarity(embeddings[i:i + 1], centroid.reshape(1, -1))[0][0])
            r["confidence"] = round(max(sim, 0.0) * 100, 2)

    return records


def save_outputs(records, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "face_index", "cluster_id", "person_label", "confidence_score"])

        for r in records:
            cid = r["cluster_id"]
            person_label = "unclustered" if cid == -1 else f"person_{cid}"
            dest_folder = output_dir / person_label
            dest_folder.mkdir(exist_ok=True)

            dest_name = f"{r['path'].stem}_face{r['face_idx']}{r['path'].suffix}"
            shutil.copy2(r["path"], dest_folder / dest_name)

            writer.writerow([r["path"].name, r["face_idx"], cid, person_label, r["confidence"]])

    print(f"\n[DONE] Results written to: {output_dir}")
    print(f"[DONE] Summary CSV: {csv_path}")


def generate_html_report(records, output_dir: Path):
    """Builds a simple, demo-friendly HTML gallery grouping images by cluster,
    with each thumbnail labeled by a confidence score and visual progress bar."""
    clusters = {}
    for r in records:
        clusters.setdefault(r["cluster_id"], []).append(r)

    def cluster_sort_key(cid):
        return (cid == -1, cid)  # unclustered (-1) sorts last

    total_faces = len(records)
    n_people = len([c for c in clusters if c != -1])
    n_unclustered = len(clusters.get(-1, []))
    overall_avg_conf = np.mean([r["confidence"] for r in records]) if records else 0.0

    sections = []
    for cid in sorted(clusters.keys(), key=cluster_sort_key):
        items = clusters[cid]
        label = "Unclustered" if cid == -1 else f"Person {cid}"
        folder = "unclustered" if cid == -1 else f"person_{cid}"
        cluster_avg_conf = np.mean([r["confidence"] for r in items])
        avg_class = "high" if cluster_avg_conf >= 70 else ("mid" if cluster_avg_conf >= 40 else "low")

        cards = []
        for r in items:
            dest_name = f"{r['path'].stem}_face{r['face_idx']}{r['path'].suffix}"
            rel_path = f"{folder}/{dest_name}"
            conf = r["confidence"]
            conf_class = "high" if conf >= 70 else ("mid" if conf >= 40 else "low")
            cards.append(f"""
            <div class="card">
              <div class="thumb"><img src="{rel_path}" alt="{dest_name}"></div>
              <div class="meta">
                <div class="meta-top">
                  <span class="fname">{r['path'].name}</span>
                  <span class="conf {conf_class}">{conf:.1f}%</span>
                </div>
                <div class="bar-track">
                  <div class="bar-fill {conf_class}" style="width:{conf:.1f}%;"></div>
                </div>
              </div>
            </div>""")

        sections.append(f"""
        <section>
          <h2>
            {label}
            <span class="count">({len(items)} image{'s' if len(items) != 1 else ''})</span>
            <span class="cluster-avg {avg_class}">avg confidence {cluster_avg_conf:.1f}%</span>
          </h2>
          <div class="grid">{''.join(cards)}</div>
        </section>""")

    stats_bar = f"""
    <div class="stats-bar">
      <div class="stat">
        <span class="stat-value">{total_faces}</span>
        <span class="stat-label">Faces Processed</span>
      </div>
      <div class="stat">
        <span class="stat-value">{n_people}</span>
        <span class="stat-label">Individuals Identified</span>
      </div>
      <div class="stat">
        <span class="stat-value">{n_unclustered}</span>
        <span class="stat-label">Unclustered</span>
      </div>
      <div class="stat">
        <span class="stat-value">{overall_avg_conf:.1f}%</span>
        <span class="stat-label">Avg Confidence</span>
      </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Face Clustering Results</title>
<style>
  * {{ box-sizing: border-box; }}

  body {{
    font-family: -apple-system, "Segoe UI", Arial, sans-serif;
    background: #0d0d0f;
    color: #eee;
    margin: 0;
    padding: 40px 48px 64px;
  }}

  h1 {{ font-size: 26px; margin: 0 0 6px; font-weight: 700; }}

  .subtitle {{
    color: #9a9a9a;
    margin-bottom: 24px;
    font-size: 14px;
    line-height: 1.5;
  }}

  .stats-bar {{
    display: flex;
    gap: 16px;
    background: #17171a;
    border: 1px solid #2a2a2e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 40px;
    flex-wrap: wrap;
  }}

  .stat {{
    flex: 1;
    min-width: 130px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .stat-value {{
    font-size: 26px;
    font-weight: 700;
    color: #fff;
  }}

  .stat-label {{
    font-size: 12px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  section {{ margin-bottom: 48px; }}

  h2 {{
    font-size: 18px;
    font-weight: 600;
    border-bottom: 1px solid #2a2a2e;
    padding-bottom: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex-wrap: wrap;
  }}

  .count {{ color: #888; font-weight: 400; font-size: 13px; }}

  .cluster-avg {{
    margin-left: auto;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 20px;
  }}

  .card {{
    background: #1a1a1d;
    border: 1px solid #2a2a2e;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}

  .card:hover {{
    transform: translateY(-3px);
    border-color: #3d3d42;
  }}

  .card .thumb {{
    width: 100%;
    aspect-ratio: 3 / 4;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}

  .card img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }}

  .meta {{
    padding: 10px 12px 12px;
    border-top: 1px solid #2a2a2e;
  }}

  .meta-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }}

  .fname {{
    color: #aaa;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
  }}

  .conf {{
    font-weight: 600;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 6px;
    white-space: nowrap;
    flex-shrink: 0;
  }}

  .conf.high {{ background: #163d1f; color: #6fe08b; }}
  .conf.mid  {{ background: #3d341a; color: #e0c56f; }}
  .conf.low  {{ background: #3d1a1a; color: #e06f6f; }}

  .cluster-avg.high {{ background: #163d1f; color: #6fe08b; }}
  .cluster-avg.mid  {{ background: #3d341a; color: #e0c56f; }}
  .cluster-avg.low  {{ background: #3d1a1a; color: #e06f6f; }}

  .bar-track {{
    width: 100%;
    height: 6px;
    background: #2a2a2e;
    border-radius: 3px;
    overflow: hidden;
  }}

  .bar-fill {{
    height: 100%;
    border-radius: 3px;
  }}

  .bar-fill.high {{ background: #6fe08b; }}
  .bar-fill.mid  {{ background: #e0c56f; }}
  .bar-fill.low  {{ background: #e06f6f; }}
</style>
</head>
<body>
  <h1>Face Identification &amp; Clustering Results</h1>
  <div class="subtitle">Each group represents one identified individual. Confidence = similarity to the cluster's identity centroid.</div>
  {stats_bar}
  {''.join(sections)}
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[DONE] Visual report: {report_path}")
    return report_path


def print_summary(records):
    clusters = {}
    for r in records:
        clusters.setdefault(r["cluster_id"], []).append(r)

    n_people = len([c for c in clusters if c != -1])
    n_unclustered = len(clusters.get(-1, []))

    print("\n===== SUMMARY =====")
    print(f"Total faces processed : {len(records)}")
    print(f"Individuals found      : {n_people}")
    print(f"Unclustered faces      : {n_unclustered}")
    for cid, items in sorted(clusters.items()):
        label = "unclustered" if cid == -1 else f"person_{cid}"
        avg_conf = np.mean([r["confidence"] for r in items])
        print(f"  {label:15s}: {len(items):3d} images | avg confidence {avg_conf:.1f}%")
    print("====================\n")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")
    print(f"[INFO] Found {len(image_paths)} images")

    records = extract_faces_and_embeddings(
        image_paths, device, keep_all_faces=args.keep_all_faces,
        margin=args.margin, image_size=args.image_size, embedding_model=args.embedding_model,
        enhance=not args.no_enhance,
    )
    if not records:
        raise SystemExit("No faces detected in any image.")

    if args.debug_distances:
        embeddings = np.stack([r["embedding"] for r in records])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dist_matrix = cosine_distances(embeddings)
        names = [f"{r['path'].stem}_f{r['face_idx']}" for r in records]

        print("\n===== PAIRWISE COSINE DISTANCE (lower = more similar) =====")
        header = " " * 22 + " ".join(f"{n[:12]:>13s}" for n in names)
        print(header)
        for i, row_name in enumerate(names):
            row = " ".join(f"{dist_matrix[i][j]:13.3f}" for j in range(len(names)))
            print(f"{row_name[:20]:22s}{row}")
        print("=============================================================\n")
        print("Pick --eps just above the largest same-person distance, "
              "and below the smallest different-person distance.")
        return

    eps = args.eps
    if args.auto_eps:
        embeddings_tmp = np.stack([r["embedding"] for r in records])
        embeddings_tmp = embeddings_tmp / np.linalg.norm(embeddings_tmp, axis=1, keepdims=True)
        dist_matrix_tmp = cosine_distances(embeddings_tmp)
        eps = auto_select_eps(dist_matrix_tmp, args.min_samples)

    records, embeddings = cluster_faces(records, eps, args.min_samples)
    records = compute_confidence_scores(records, embeddings)

    save_outputs(records, output_dir)
    print_summary(records)

    if args.report:
        report_path = generate_html_report(records, output_dir)
        webbrowser.open(report_path.resolve().as_uri())


if __name__ == "__main__":
    main()