"""Compare YOLOv5 plate-detector weights on a demo video."""

import argparse
import csv
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_WEIGHTS = [
    "data/weights/skud.pt",
]


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def load_yolov5_model(path: Path, device: str, conf: float, iou: float):
    capture = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(path),
                source="github",
                trust_repo=True,
                verbose=False,
            )
    model.conf = conf
    model.iou = iou
    model.max_det = 20
    torch_device = "cuda:0" if device == "0" else device
    model.to(torch_device)
    return model


def predict(model, frame, image_size: int):
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model(frame, size=image_size)

    boxes = results.xyxy[0]
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for row in boxes.detach().cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = row[:6]
        detections.append(
            {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
                "class_id": int(cls_id),
            }
        )
    return detections


def draw_detections(frame, detections, names, title):
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        image,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = names.get(det["class_id"], str(det["class_id"])) if isinstance(names, dict) else str(det["class_id"])
        label = f"{class_name} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
    return image


def read_sample_frames(video_path: str, frame_numbers):
    frames = {}
    wanted = set(frame_numbers)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened() and wanted:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx in wanted:
            frames[frame_idx] = frame
            wanted.remove(frame_idx)
    cap.release()
    return frames


def make_contact_sheet(image_paths, output_path: Path, columns: int = 2):
    images = [cv2.imread(str(path)) for path in image_paths if path.exists()]
    images = [image for image in images if image is not None]
    if not images:
        return

    thumb_w = 640
    thumb_h = int(images[0].shape[0] * (thumb_w / images[0].shape[1]))
    thumbs = [cv2.resize(image, (thumb_w, thumb_h)) for image in images]

    rows = int(np.ceil(len(thumbs) / columns))
    sheet = np.full((rows * thumb_h, columns * thumb_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        y = row * thumb_h
        x = col * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def evaluate_weight(path: Path, args, sample_frames):
    result = {
        "weight": str(path),
        "loaded": False,
        "names": "",
        "hit_frames": 0,
        "detections": 0,
        "avg_conf": 0.0,
        "max_conf": 0.0,
        "error": "",
    }

    try:
        model = load_yolov5_model(path, args.device, args.conf, args.iou)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return result, []

    names = getattr(model, "names", {})
    result["loaded"] = True
    result["names"] = str(names)

    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    confs = []
    saved_images = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break
        if frame_idx % args.step != 0:
            continue

        detections = predict(model, frame, args.imgsz)
        if detections:
            result["hit_frames"] += 1
            result["detections"] += len(detections)
            confs.extend(det["confidence"] for det in detections)

    cap.release()

    if confs:
        result["avg_conf"] = sum(confs) / len(confs)
        result["max_conf"] = max(confs)

    for sample_idx, frame in sample_frames.items():
        detections = predict(model, frame, args.imgsz)
        annotated = draw_detections(
            frame,
            detections,
            names,
            f"{path.name} | frame {sample_idx} | {len(detections)} det",
        )
        image_path = args.output_dir / f"{safe_name(path)}_frame_{sample_idx}.jpg"
        cv2.imwrite(str(image_path), annotated)
        saved_images.append(image_path)

    return result, saved_images


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv5 detector weights")
    parser.add_argument("--video", default="data/videos/demo.mp4")
    parser.add_argument("--weights", nargs="*", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_tests"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--sample-frames", default="15,75,120,180,300")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = [Path(weight).resolve() for weight in args.weights]
    sample_numbers = [int(value) for value in args.sample_frames.split(",") if value.strip()]
    sample_frames = read_sample_frames(args.video, sample_numbers)

    rows = []
    images_by_frame = {frame_idx: [] for frame_idx in sample_frames}
    for weight in weights:
        print(f"Testing {weight.name}...")
        row, image_paths = evaluate_weight(weight, args, sample_frames)
        rows.append(row)
        for image_path in image_paths:
            match = re.search(r"_frame_(\d+)\.jpg$", image_path.name)
            if match:
                images_by_frame[int(match.group(1))].append(image_path)

    csv_path = args.output_dir / "yolov5_weight_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for frame_idx, image_paths in images_by_frame.items():
        make_contact_sheet(
            image_paths,
            args.output_dir / f"comparison_frame_{frame_idx}.jpg",
        )

    print("\nYOLOv5 weight comparison")
    print("=" * 90)
    print(f"{'weight':22} {'loaded':7} {'hit':>5} {'det':>5} {'avg':>6} {'max':>6} names/error")
    for row in rows:
        status = "yes" if row["loaded"] else "no"
        info = row["names"] if row["loaded"] else row["error"]
        print(
            f"{Path(row['weight']).name[:22]:22} {status:7} "
            f"{row['hit_frames']:5} {row['detections']:5} "
            f"{row['avg_conf']:.3f} {row['max_conf']:.3f} {info}"
        )

    print("\nSaved:")
    print(f"- {csv_path.resolve()}")
    print(f"- {args.output_dir.resolve()}/*frame*.jpg")


if __name__ == "__main__":
    main()
