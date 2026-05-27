from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from correct_paper import correct_paper


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def iter_sample_images(sample_dir: Path) -> list[Path]:
    images = []
    for path in sorted(sample_dir.iterdir()):
        if path.suffix.lower() not in VALID_SUFFIXES:
            continue
        if "corrected" in path.stem.lower():
            continue
        images.append(path)
    return images


def process_one(input_path: Path, output_path: Path) -> float:
    image_bytes = input_path.read_bytes()
    start = perf_counter()
    corrected = correct_paper(image_bytes)
    elapsed = perf_counter() - start
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(corrected)
    return elapsed


def _process_one_worker(input_path: str, output_path: str, queue: mp.Queue) -> None:
    try:
        elapsed = process_one(Path(input_path), Path(output_path))
        queue.put(("ok", elapsed))
    except Exception:
        queue.put(("err", traceback.format_exc()))


def process_one_isolated(input_path: Path, output_path: Path) -> float:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_process_one_worker,
        args=(str(input_path), str(output_path), queue),
    )
    process.start()
    result = queue.get()
    process.join()

    if process.exitcode != 0 and result[0] != "err":
        raise RuntimeError(f"worker exited with code {process.exitcode}")
    if result[0] == "err":
        raise RuntimeError(result[1])
    return float(result[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process images in samples/")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional single image path. If omitted, process all images in samples/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/batch_samples"),
        help="Directory for batch outputs.",
    )
    args = parser.parse_args()

    if args.input is not None:
        suffix = args.input.suffix.lower() or ".jpg"
        output_path = args.output_dir / f"{args.input.stem}_corrected{suffix}"
        elapsed = process_one(args.input, output_path)
        print(f"saved: {output_path}")
        print(f"elapsed: {elapsed:.4f}s")
        return

    sample_dir = Path("samples")
    images = iter_sample_images(sample_dir)
    if not images:
        print("No sample images found.")
        return

    total = 0.0
    for path in images:
        output_path = args.output_dir / path.name
        elapsed = process_one_isolated(path, output_path)
        total += elapsed
        print(f"{path.name}: {elapsed:.4f}s -> {output_path}")

    print(f"processed: {len(images)} images")
    print(f"avg elapsed: {total / len(images):.4f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
