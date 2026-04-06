"""
Script to filter images in cam1-cam4 subfolders based on time ranges
defined in field_time_ranges.csv.

Image filename format: axis-{cam}_{seq}_{YYYY-MM-DD}_{HH-MM-SS}.{ext}
The timestamp is extracted, compared against CSV bins, and matching
filenames are recorded to a txt file per field.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent / "data/soilsense/september/"
CSV_PATH = BASE_DIR / "field_time_ranges.csv"
CAM_FOLDERS = ["2", "3", "4", "5"]
OUTPUT_FILE = BASE_DIR / "matched_images.txt"


def parse_timestamp_from_filename(filename: str) -> datetime | None:
    """Extract UTC datetime from an image filename like axis-1_00001_2025-10-22_10-49-50.png"""
    stem = Path(filename).stem  # strip extension
    parts = stem.split("_")
    # parts: ['axis-1', '00001', '2025-10-22', '10-49-50']
    if len(parts) < 4:
        return None
    try:
        date_str = parts[2]           # '2025-10-22'
        time_str = parts[3]           # '10-49-50'
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def load_time_ranges(csv_path: str) -> list[dict]:
    """Load field time ranges from CSV."""
    ranges = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field_name = row["field_name"].strip()
            start = datetime.fromisoformat(row["start_time"].strip())
            end = datetime.fromisoformat(row["end_time"].strip())
            ranges.append({"field_name": field_name, "start": start, "end": end})
    return ranges


def main():
    time_ranges = load_time_ranges(CSV_PATH)
    print(f"Loaded {len(time_ranges)} time ranges from CSV:")
    for tr in time_ranges:
        print(f"  {tr['field_name']}: {tr['start']} -> {tr['end']}")

    # Collect matches: {field_name: [list of (cam_folder, filename)]}
    matches = {tr["field_name"]: [] for tr in time_ranges}
    unmatched = []
    total_images = 0

    for cam in CAM_FOLDERS:
        cam_path = BASE_DIR / cam
        if not cam_path.is_dir():
            print(f"Warning: {cam_path} not found, skipping.")
            continue

        filenames = sorted(os.listdir(cam_path))
        for fname in filenames:
            total_images += 1
            ts = parse_timestamp_from_filename(fname)
            if ts is None:
                continue

            matched = False
            for tr in time_ranges:
                if tr["start"] <= ts <= tr["end"]:
                    # Verify image is valid with PIL before adding
                    try:
                        img = Image.open(cam_path / fname)
                        img.verify()
                    except Exception:
                        print(f"  Skipping corrupt image: {cam}/{fname}")
                        break
                    matches[tr["field_name"]].append((cam, fname))
                    matched = True
                    break  # assign to first matching bin only

            if not matched:
                unmatched.append((cam, fname))

    # Write all matched images to a single txt file (subfolder/image_name)
    with open(OUTPUT_FILE, "w") as f:
        for tr in time_ranges:
            field = tr["field_name"]
            imgs = matches[field]
            for cam, fname in imgs:
                f.write(f"{cam}/{fname}\n")
            print(f"  {field}: {len(imgs)} images")

    total_matched = sum(len(v) for v in matches.values())
    print(f"\nProcessed {total_images} images across {len(CAM_FOLDERS)} cameras.")
    print(f"Matched: {total_matched}, Unmatched: {len(unmatched)}")


if __name__ == "__main__":
    main()
