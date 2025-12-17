import argparse
from pathlib import Path
import numpy as np
import os

from predict import main as predict_main
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

DEFAULT_ROOT = Path(
    r"/workspace/Ratatouille/ingredients"
    r"/HeadFixedTraining/SensoryPrediction_202512"
)
DEFAULT_MODEL = Path(
    # r"/workspace/meye/models/meye-2022-01-24.h5"
    r"/workspace/meye/models/meye-2025-03-12-Offline.h5"
)


def result_save_path(video_path: Path) -> Path:
    """Build output CSV path following the VIDEO_... naming convention."""
    video_path = Path(video_path)
    dir_name, file_name = os.path.split(video_path)
    if not (file_name.startswith("VIDEO_") and file_name.endswith(".mp4")):
        raise ValueError(f"Unexpected video name {file_name}, expected VIDEO_*.mp4")

    session_name = file_name[len("VIDEO_") : -len(".mp4")]
    indent = ".." if os.path.basename(dir_name) == "video" else ""
    save_path = os.path.join(dir_name, indent, "pupil", f"PUPIL_{session_name}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return Path(save_path)


def corresponding_video_path(npz_path: Path):
    return Path(str(npz_path).replace("EYE_", "VIDEO_")).with_suffix(".mp4")


def _load_metadata(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    try:
        bbox = data["bbox"]
    except KeyError as exc:
        raise RuntimeError(f"Missing expected field {exc} in {npz_path}") from exc

    if len(bbox) != 4:
        raise RuntimeError(f"bbox in {npz_path} must have 4 elements, got {len(bbox)}")

    x, y, w, h = [int(v) for v in bbox]
    rl, rt = x, y
    rr, rb = x + w, y + h
    video_path = corresponding_video_path(npz_path)
    return (rl, rt, rr, rb), video_path


def run_predictions(root, model, thr=0.5, skip_existing=False):
    npz_files = sorted(Path(root).rglob("*EYE_*.npz"))
    print(f"Found {len(npz_files)} .npz files")
    for npz_idx, npz_path in enumerate(npz_files):
        if not npz_path.name.startswith("EYE_"):
            continue

        (rl, rt, rr, rb), video_path = _load_metadata(npz_path)

        if not video_path.exists():
            raise FileNotFoundError(f"[SKIP] {npz_path}: video not found at {video_path}")

        output_csv = result_save_path(video_path)
        output_video = output_csv.with_suffix(".mp4")

        if skip_existing and output_csv.exists():  #  and output_video.exists()
            print(f"[SKIP] outputs already exist for {video_path}")
            continue

        print(f"[RUN ] {npz_path.name} -> {video_path}")
        args = argparse.Namespace(
            model=str(model),
            video=str(video_path),
            thr=float(thr),
            rl=rl,
            rt=rt,
            rr=rr,
            rb=rb,
            output_video=str(output_video) if npz_idx % 5 == 1 else None,
            output_csv=str(output_csv),
        )

        try:
            predict_main(args)
            print(f"[DONE] wrote {output_video} and {output_csv}")
        except Exception as exc:
            print(f"[FAIL] {video_path}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch predict videos described by EYE_*.npz files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan for .npz files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the Keras model to use.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Threshold forwarded to predict.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing when both outputs already exist.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}")

    run_predictions(args.root, args.model, thr=args.thr, skip_existing=args.skip_existing)

"""
docker run --gpus all -it --rm -v "$(Split-Path -Parent $PWD):/workspace" -w "/workspace/$(Split-Path -Leaf $PWD)" meye bash
"""
