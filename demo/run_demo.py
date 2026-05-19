"""
Run the LPR pipeline on a short video and write an annotated MP4.
"""

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.setup_whitelist import setup_whitelist
from main import LPRSystem
from src.utils.config import Config
from src.utils.logger import logger


def count_access_logs(db_path: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM access_log")
        return cursor.fetchone()[0]


def run_demo(
    config_path: str,
    video_path: str,
    output_path: str,
    seconds: float,
    process_every: int,
    max_frames: int = None,
) -> int:
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)

    config = Config(config_path)
    setup_whitelist()

    before_logs = count_access_logs(config.database.path)
    system = LPRSystem(config_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is None:
        max_frames = total_frames if seconds <= 0 else int(seconds * fps)
    if total_frames > 0:
        max_frames = min(max_frames, total_frames)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames = 0
    analyzed_frames = 0
    detections = 0
    recognized = []
    granted = []
    denied = []
    last_result = {
        "vehicles": [],
        "plates": [],
        "matches": [],
        "recognized_plates": [],
        "results": [],
    }

    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frames += 1
            if (frames - 1) % process_every == 0:
                last_result = system.pipeline.process_frame(frame)
                analyzed_frames += 1

                detections += len(last_result["plates"])
                for plate in last_result["recognized_plates"]:
                    raw_text = plate.get("raw_text", plate["text"])
                    recognized.append((raw_text, plate["text"]))

                for access in last_result["results"]:
                    entry = (access["plate"], access["reason"])
                    if access["authorized"]:
                        granted.append(entry)
                    else:
                        denied.append(entry)

            writer.write(system._visualize_results(frame, last_result))
    finally:
        cap.release()
        writer.release()

    after_logs = count_access_logs(config.database.path)
    unique_granted = sorted({plate for plate, _ in granted})
    unique_denied = sorted({plate for plate, _ in denied})
    unique_recognized = sorted({resolved for _, resolved in recognized})

    print("\nDemo result")
    print("=" * 60)
    print(f"Video:              {video_path}")
    print(f"Frames written:     {frames}")
    print(f"Frames analyzed:    {analyzed_frames}")
    print(f"Process every:      {process_every} frame(s)")
    print(f"Output duration:    {frames / fps:.1f}s")
    print(f"Plate detections:   {detections}")
    print(f"Recognized plates:  {', '.join(unique_recognized) or '-'}")
    print(f"Access granted:     {', '.join(unique_granted) or '-'}")
    print(f"Access denied:      {', '.join(unique_denied) or '-'}")
    print(f"DB log rows added:  {after_logs - before_logs}")
    print(f"Annotated output:   {output.resolve()}")

    passed = bool(unique_granted) and output.exists() and output.stat().st_size > 0
    print(f"Status:             {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="Run LPR video demo")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video", default="data/videos/demo.mp4")
    parser.add_argument("--output", default="data/videos/demo_output.mp4")
    parser.add_argument(
        "--seconds",
        type=float,
        default=6,
        help="How many seconds of video to process. Use 0 for the full video.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=30,
        help="Run detection/OCR every N frames while still writing every frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Frame limit override. Usually --seconds is easier.",
    )
    args = parser.parse_args()

    raise SystemExit(
        run_demo(
            config_path=args.config,
            video_path=args.video,
            output_path=args.output,
            seconds=args.seconds,
            process_every=max(1, args.process_every),
            max_frames=args.max_frames,
        )
    )


if __name__ == "__main__":
    main()
