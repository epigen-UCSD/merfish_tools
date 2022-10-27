import sys
import pathlib
import re
from collections import namedtuple
from tqdm import tqdm
import pandas as pd

ImageFile = namedtuple("ImageFile", ["round", "fov", "size"])


def verify_folder(folder):
    files = []
    for dax in tqdm(list(folder.glob("*.dax")), desc=f"Checking {folder}"):
        size = dax.stat().st_size
        match = re.search("_H([0-9]+)_F_?([0-9]+).dax", str(dax))
        round = int(match.group(1))
        fov = int(match.group(2))
        files.append(ImageFile(round, fov, size))

    if not files:
        print(f"FAIL: No dax files found in folder. Is it the correct path?")
        return

    df = pd.DataFrame(files)

    n_rounds = len(df["round"].unique())
    n_fovs = len(df["fov"].unique())
    n_files = len(df)

    print(f"INFO: Found images from {n_rounds} imaging rounds and {n_fovs} FOVs")

    if n_files != n_rounds * n_fovs:
        print(
            f"FAIL: Found {n_files} images ({n_rounds} * {n_fovs} = {n_rounds * n_fovs})"
        )
        fovs = df["fov"].unique()
        for hyb in df["round"].unique():
            hfovs = df[df["round"] == hyb]["fov"].unique()
            missing = set(fovs) - set(hfovs)
            if missing:
                print(f"Hyb {hyb}, FOVs: {missing}")
    else:
        print(
            f"PASS: Found {n_files} images ({n_rounds} * {n_fovs} = {n_rounds * n_fovs})"
        )

    for round in df.groupby("round")["size"].unique():
        if len(round) > 1:
            print("FAIL: Some images in the same round are different sizes")
            break
    else:
        print("PASS: All images in the same round are the same size")


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    verify_folder(folder)
