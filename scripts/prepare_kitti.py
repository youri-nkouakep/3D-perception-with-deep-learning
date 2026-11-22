"""Download what is needed from KITTI 3D object and keep only the LiDAR points seen by the camera.

The 27 GB velodyne archive is never stored: each scan is read straight out of the remote zip
with HTTP range requests, cropped and saved as float16 (about 1.5 GB in total).
Can be stopped and run again, finished scans are skipped.

    python scripts/prepare_kitti.py --out data/processed/kitti
"""

import argparse
import io
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from perception.datasets.kitti import KITTI_URL, Calib, HttpRangeFile, crop_to_camera, read_split

SPLITS_URL = "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def range_get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except OSError:
            if attempt == 4:
                raise


def read_member(url: str, info: zipfile.ZipInfo) -> bytes:
    # the local header is 30 bytes + name + extra field, the extra length is only known once read
    margin = 30 + len(info.filename) + 256
    blob = range_get(url, info.header_offset, info.header_offset + margin + info.compress_size)
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    raw = blob[start:start + info.compress_size]
    if len(raw) < info.compress_size:
        raw = range_get(url, info.header_offset + start, info.header_offset + start + info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    return raw


def extract_small_zip(name: str, folder: str, out: Path) -> None:
    target = out / folder
    if target.exists() and any(target.iterdir()):
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(fetch(KITTI_URL + name))) as z:
        for member in z.namelist():
            if member.startswith(f"training/{folder}/") and member.endswith(".txt"):
                (target / Path(member).name).write_bytes(z.read(member))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed/kitti"))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    out = args.out

    (out / "splits").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        f = out / "splits" / f"{split}.txt"
        if not f.exists():
            f.write_bytes(fetch(SPLITS_URL + f"{split}.txt"))
    extract_small_zip("data_object_label_2.zip", "label_2", out)
    extract_small_zip("data_object_calib.zip", "calib", out)
    print("labels, calibration and splits ready")

    ids = read_split(out / "splits" / "train.txt") + read_split(out / "splits" / "val.txt")
    (out / "points").mkdir(exist_ok=True)
    todo = [i for i in ids if not (out / "points" / f"{i}.npy").exists()]
    chunks = [todo[k::args.workers] for k in range(args.workers)]
    bar = tqdm(total=len(ids), initial=len(ids) - len(todo), desc="LiDAR scans")

    url = KITTI_URL + "data_object_velodyne.zip"
    # read the zip index once, then fetch each scan with a single range request
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)) as z:
        members = {Path(i.filename).stem: i for i in z.infolist() if i.filename.startswith("training/velodyne/")}

    def work(chunk: list[str]) -> None:
        for idx in chunk:
            pts = np.frombuffer(read_member(url, members[idx]), dtype=np.float32).reshape(-1, 4)
            calib = Calib.from_text((out / "calib" / f"{idx}.txt").read_text())
            pts = crop_to_camera(pts, calib)
            # write then rename, so a stopped run never leaves a half-written scan
            tmp = out / "points" / f"{idx}.part.npy"
            np.save(tmp, pts.astype(np.float16))
            tmp.replace(out / "points" / f"{idx}.npy")
            bar.update()

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, chunks))
    bar.close()


if __name__ == "__main__":
    main()
