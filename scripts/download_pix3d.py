#!/usr/bin/env python3
"""
Download the standard Pix3D dataset and extract only the cleaned 3D models
(.obj, .mtl, and texture maps) from model/**, skipping images/annotations.

Resumable: skips existing files by default. Use --overwrite to force.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PIX3D_URL = "http://pix3d.csail.mit.edu/data/pix3d.zip"

# Keep only meshes, materials, and common texture formats
KEEP_EXTS = {".obj", ".mtl", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff"}


def log(msg):  # tiny logger
    print(f"[INFO] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def err(msg):
    print(f"[ERR]  {msg}", file=sys.stderr)


def should_keep(member: str) -> bool:
    """Keep only files under model/ with allowed extensions."""
    member = member.replace("\\", "/")
    if member.endswith("/"):
        return False  # skip pure directories (we'll create parents as needed)
    if not member.startswith("model/"):
        return False
    return Path(member).suffix.lower() in KEEP_EXTS


def is_within_directory(base: Path, target: Path) -> bool:
    """Prevent path traversal when extracting."""
    try:
        base = base.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        return False
    return str(target).startswith(str(base))


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading Pix3D (~3.6 GB): {url}")
    urlretrieve(url, dst)
    log(f"Download complete: {dst}")


def extract_models_only(zip_path: Path, out_dir: Path, overwrite: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting cleaned models from {zip_path} to {out_dir} ...")
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for zi in zf.infolist():
            name = zi.filename.replace("\\", "/")
            if not should_keep(name):
                continue

            target = out_dir / name
            # zip-slip guard
            if not is_within_directory(out_dir, target):
                warn(f"Skipping suspicious path: {name}")
                continue

            # Ensure parent dirs
            os.makedirs(target.parent, exist_ok=True)

            # Resume logic: skip if exists with same size (best-effort)
            if target.exists() and not overwrite:
                # Compare sizes if it’s a regular file
                try:
                    with zf.open(zi, "r") as src:
                        zsize = zi.file_size
                    fsize = target.stat().st_size
                    if fsize == zsize:
                        skipped += 1
                        continue
                except Exception:
                    # If any check fails, fall back to skipping to be safe
                    skipped += 1
                    continue

            # Extract
            with zf.open(zi, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    log(f"Done. Extracted {extracted} files. Skipped {skipped} existing files.")


def main():
    ap = argparse.ArgumentParser(description="Download Pix3D cleaned dataset (models only).")
    ap.add_argument("output_dir", help="Destination directory for extracted 3D models")
    ap.add_argument("--keep-archive", action="store_true", help="Keep the downloaded zip file")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files even if they exist")
    ap.add_argument("--zip-path", default=None,
                    help="Use an existing pix3d.zip instead of downloading")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        zip_path = Path(args.zip_path).resolve()
        if not zip_path.exists():
            err(f"--zip-path does not exist: {zip_path}")
            sys.exit(1)
        log(f"Using existing archive: {zip_path}")
    else:
        zip_path = out_dir / "pix3d.zip"
        if not zip_path.exists():
            download(PIX3D_URL, zip_path)
        else:
            log(f"Archive already exists: {zip_path}")

    extract_models_only(zip_path, out_dir, overwrite=args.overwrite)

    if not args.zip_path and not args.keep_archive:
        try:
            zip_path.unlink()
            log(f"Deleted archive {zip_path}")
        except Exception as e:
            warn(f"Could not delete archive: {e}")


if __name__ == "__main__":
    main()
