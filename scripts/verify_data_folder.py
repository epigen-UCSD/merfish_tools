import sys
import pathlib
import re
from collections import namedtuple
import pandas as pd

ImageFile = namedtuple("ImageFile", ["round", "fov", "size"])


def parse_image_filename(filename):
    # Try homebuilt microscope filename
    match = re.search("_H([0-9]+)_F_?([0-9]+).dax", filename)
    if match:
        hyb = int(match.group(1))
        fov = int(match.group(2))
        return hyb, fov

    # Try MERSCOPE filename
    match = re.search("stack_([prestain_0-9]+)_([0-9]+).dax", filename)
    if match:
        hyb = match.group(1)
        fov = int(match.group(2))
        return hyb, fov

    raise AttributeError(f"Can't determine FOV and imaging round for filename {filename}")


def verify_folder(folder):
    files = []
    print(f"Checking {folder}...")

    filenames = list(folder.glob("*.dax"))
    if not filenames:
        filenames = list(pathlib.Path(folder / "data").glob("*.dax"))
        if not filenames:
            print("FAIL: Could not find any image files. Is it the correct path?")
            sys.exit(1)
        fovs = len(pathlib.Path(folder / "settings" / "positions.csv").read_text(encoding="utf-8").split())
        print(f"Found MERSCOPE positions file with {fovs} FOVs")

    for dax in filenames:
        size = dax.stat().st_size
        hyb, fov = parse_image_filename(str(dax))
        files.append(ImageFile(hyb, fov, size))

    df = pd.DataFrame(files)

    n_rounds = len(df["round"].unique())
    n_fovs = len(df["fov"].unique())
    n_files = len(df)

    print(f"Found images from {n_rounds} imaging rounds and {n_fovs} FOVs")

    if n_files != n_rounds * n_fovs:
        print(f"FAIL: Found {n_files} images ({n_rounds} * {n_fovs} = {n_rounds * n_fovs})")
        fovs = df["fov"].unique()
        for hyb in df["round"].unique():
            hfovs = df[df["round"] == hyb]["fov"].unique()
            missing = set(fovs) - set(hfovs)
            if missing:
                print(f"Hyb {hyb}, FOVs: {missing}")
    else:
        print(f"PASS: Found {n_files} images ({n_rounds} * {n_fovs} = {n_rounds * n_fovs})")

    for hyb in df.groupby("round")["size"].unique():
        if len(hyb) > 1:
            print("FAIL: Some images in the same round are different sizes")
            break
    else:
        print("PASS: All images in the same round are the same size")


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    verify_folder(folder)
