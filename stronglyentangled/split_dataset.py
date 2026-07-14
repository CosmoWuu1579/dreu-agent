"""
Split a Br35H-style dataset into the train/val/test layout the paper's CNN-*.py
scripts expect.

INPUT  (Kaggle Br35H): a folder with two subfolders of images --
    <source>/yes/*.jpg|png     (tumor,   label 1)
    <source>/no/*.jpg|png      (healthy, label 0)

OUTPUT (what CNN-*.py loads from ./data/Br35H):
    <out>/train/{yes,no}/ ...
    <out>/val/{yes,no}/   ...
    <out>/test/{yes,no}/  ...

By default it COPIES a SUBSET (the paper's 1000/200/400 total = 500/100/200 per
class) into a NEW folder, so the original Kaggle folder is untouched and can be
deleted afterward to reclaim space. Because it only copies the subset you asked
for, <out> is smaller than the full 3000-image download.

    # copy the paper-sized subset into ./data/Br35H, leave the Kaggle folder alone
    python split_dataset.py  path/to/kaggle_br35h  --out ./data/Br35H

    # tight on disk? --move relocates files instead of copying (no duplication;
    # empties the source as it goes, so there is nothing left to delete)
    python split_dataset.py  path/to/kaggle_br35h  --out ./data/Br35H --move

    # use MORE of the 1500/class you have (e.g. a 1000/200/300 per-class split)
    python split_dataset.py  SRC --out ./data/Br35H --train 1000 --val 200 --test 300

Only the standard library is used (os, shutil, random) -- no extra packages.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

CLASSES = ("yes", "no")
EXTS = (".png", ".jpg", ".jpeg")


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def list_images(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(EXTS))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Split a Br35H yes/no folder into train/val/test/{yes,no}.")
    p.add_argument("source", help="folder containing yes/ and no/ subfolders")
    p.add_argument("--out", default="./data/Br35H",
                   help="output root (default: ./data/Br35H, where CNN-*.py looks)")
    p.add_argument("--train", type=int, default=500, help="images PER CLASS for train (default 500)")
    p.add_argument("--val", type=int, default=100, help="images PER CLASS for val (default 100)")
    p.add_argument("--test", type=int, default=200, help="images PER CLASS for test (default 200)")
    p.add_argument("--move", action="store_true",
                   help="move files instead of copying (no duplication; consumes the source)")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed (default 42)")
    p.add_argument("--force", action="store_true",
                   help="allow writing into a non-empty --out folder")
    args = p.parse_args()

    source = os.path.abspath(args.source)
    out = os.path.abspath(args.out)

    # --- validate source layout ---
    missing = [c for c in CLASSES if not os.path.isdir(os.path.join(source, c))]
    if missing:
        print(f"ERROR: {source!r} is missing subfolder(s): {missing}. "
              f"Expected the Kaggle Br35H layout: <source>/yes/ and <source>/no/.")
        return 2
    if os.path.abspath(out) == source:
        print("ERROR: --out must differ from the source folder.")
        return 2
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        print(f"ERROR: --out {out!r} is not empty. Delete it or pass --force.")
        return 2

    per_class_need = args.train + args.val + args.test
    splits = (("train", args.train), ("val", args.val), ("test", args.test))
    op = shutil.move if args.move else shutil.copy2
    verb = "moving" if args.move else "copying"
    rng = random.Random(args.seed)

    print(f"source : {source}")
    print(f"out    : {out}")
    print(f"mode   : {verb} | per-class: train={args.train} val={args.val} "
          f"test={args.test} (need {per_class_need}/class)\n")

    copied_bytes = 0
    grand_total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(source, cls)
        files = list_images(cls_dir)
        if len(files) < per_class_need:
            print(f"ERROR: class {cls!r} has only {len(files)} images but "
                  f"{per_class_need} were requested. Lower --train/--val/--test.")
            return 2
        rng.shuffle(files)
        idx = 0
        for split, count in splits:
            dest_dir = os.path.join(out, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            chosen = files[idx:idx + count]
            idx += count
            for fname in chosen:
                src_path = os.path.join(cls_dir, fname)
                copied_bytes += os.path.getsize(src_path)
                op(src_path, os.path.join(dest_dir, fname))
            grand_total += count
            print(f"  {split:5s}/{cls:3s}: {count} images")

    past = "moved" if args.move else "copied"
    print(f"\nDone: {grand_total} images {past} "
          f"({human(copied_bytes)} in {out}).")
    if args.move:
        print(f"Source images were MOVED, so {source} now holds only the leftover "
              f"(unused) images -- delete it whenever you like.")
    else:
        print(f"The original {source} is UNTOUCHED and no longer needed -- delete "
              f"it to reclaim ~{human(directory_size(source))}.")
    print("\nPoint the CNN scripts at this data: they load ./data/Br35H, so run "
          "them from the folder that contains your --out, or set --out to "
          "<StronglyEntangled>/data/Br35H.")
    return 0


def directory_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


if __name__ == "__main__":
    sys.exit(main())
