import os
import re
import glob
import json
import pickle
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def search_for_mask_file(segmask_dir, fov):
    patterns = [
        # Bogdan's segmentation script
        f"Fov-0*{fov}_seg.pkl",
        # Cellpose numpy output
        # We prefer the npy file so we don't need the PIL library
        f"Conv_zscan_H0_F_0*{fov}_seg.npy",  # Homebuilt microscope filenames
        f"stack_prestain_0*{fov}_seg.npy",  # MERSCOPE filenames
        # Try the png cellpose output if the numpy files aren't there
        f"stack_prestain_0*{fov}_cp_masks.png",
        f"Conv_zscan_H0_F_0*{fov}_cp_masks.png",
    ]
    for filename in Path(segmask_dir).glob(f"*{fov}*"):
        for pattern in patterns:
            if re.search(pattern, str(filename)):
                return filename
    raise Exception(f"No mask found in {segmask_dir} for FOV {fov}")


def load_mask(segmask_dir: str, fov: int) -> np.ndarray:
    filename = str(search_for_mask_file(segmask_dir, fov))

    if filename.endswith(".pkl"):
        pkl = pickle.load(open(filename, "rb"))
        return pkl[0].astype(np.uint32)
    elif filename.endswith(".npy"):
        return np.load(filename, allow_pickle=True).item()["masks"]
    elif filename.endswith(".png"):
        from PIL import Image

        return np.asarray(Image.open(filename))

    raise Exception(f"Unknown format for mask {filename}")


def load_all_masks(segmask_dir: str, n_fovs: int) -> list:
    return [
        load_mask(segmask_dir, fov)
        for fov in tqdm(range(n_fovs), desc="Loading cell masks")
    ]


def save_cell_links(links, filename):
    with open(filename, "w") as f:
        for link in links:
            print(repr(link), file=f)


def load_cell_links(filename):
    links = []
    with open(filename) as f:
        for line in f:
            links.append(eval(line))
    return links


def save_barcode_table(barcodes, filename):
    barcodes.to_csv(filename, index=False)


def load_barcode_table(filename):
    return pd.read_csv(filename)


def save_cell_metadata(celldata, filename):
    celldata.to_csv(filename)


def load_cell_metadata(filename):
    return pd.read_csv(filename, index_col=0)


def save_cell_by_gene_table(cellbygene, filename):
    cellbygene.to_csv(filename)


def load_cell_by_gene_table(filename):
    return pd.read_csv(filename, index_col=0)


def save_stats(stats, filename) -> None:
    text = json.dumps(stats, indent=4)
    with open(filename, "w") as f:
        f.write(text)


def load_stats(filename):
    return json.load(open(filename))


class MerlinOutput:
    """A class for loading results from a MERlin output folder."""

    def __init__(self, folderpath):
        self.root = Path(folderpath)

    def n_fovs(self):
        path = self.root / "Decode" / "barcodes"
        return len(list(path.glob("barcode_data_*.h5")))

    def load_raw_barcodes(self, fov):
        """Load detailed barcode metadata from the Decode folder."""
        path = self.root / "Decode" / "barcodes" / f"barcode_data_{fov}.h5"
        return pd.read_hdf(path)

    def count_raw_barcodes(self, fov):
        """Count the number of barcodes for an fov in the Decode folder."""
        path = self.root / "Decode" / "barcodes" / f"barcode_data_{fov}.h5"
        barcodes = h5py.File(path, "r")
        return len(barcodes["barcodes/table"])

    def load_filtered_barcodes(self, fov):
        """Load detailed barcode metadata from the AdaptiveFilterBarcodes folder."""
        path = (
            self.root / "AdaptiveFilterBarcodes" / "barcodes" / f"barcode_data_{fov}.h5"
        )
        return pd.read_hdf(path)

    def load_exported_barcodes(self):
        """Load the exported barcode table."""
        path = self.root / "ExportBarcodes" / "barcodes.csv"
        return pd.read_csv(path)

    def load_drift_transformations(self, fov: int) -> np.ndarray:
        """Get the drifts calculated between hybridization rounds for the given FOV.

        Returns a numpy array containing scikit-image SimilarityTransform objects.
        The load_hyb_drifts function can be used instead to convert this to a
        pandas DataFrame.
        """
        possible_dirs = ["FiducialBeadWarp", "FiducialCorrelationWarp"]
        for dir in possible_dirs:
            if Path(self.root, dir).exists():
                path = self.root / dir / "transformations" / f"offsets_{fov}.npy"
                break
        else:
            return None  # TODO: Error message
        return np.load(path, allow_pickle=True)

    def load_hyb_drifts(self, fov: int) -> pd.DataFrame:
        """Get the drifts calculated between hybridization rounds for the given FOV.

        The 'X drift' and 'Y drift' columns indicate the translation required to
        align coordinates in the FOV and hybridization round to the first hybridization
        round for that FOV. These drifts are calculated by MERlin.
        """
        rows = []
        drifts = self.load_drift_transformations(fov)
        for bit, drift in enumerate(drifts, start=1):
            rows.append([fov, bit, drift.params[0][2], drift.params[1][2]])
        return pd.DataFrame(rows, columns=["FOV", "Bit", "X drift", "Y drift"])

    def load_codebook(self) -> pd.DataFrame:
        """Get the codebook used for this MERFISH experiment.

        The 'name' and 'id' columns are identical, and both contain the name of the
        gene or blank barcode encoded by that row. The 'bit1' through 'bitN' columns
        contain the 0s or 1s of the barcode.
        """
        return pd.read_csv(list(self.root.glob("codebook_*.csv"))[0])

    def load_fov_positions(self) -> pd.DataFrame:
        """Get the global positions of the FOVs.

        The coordinates indicate the top-left corner of the FOV.
        """
        path = self.root / "positions.csv"
        df = pd.read_csv(path, header=None)
        df.columns = ["x", "y"]
        return df


class DaxFile:
    def __init__(self, filename, num_channels=None):
        self.filename = filename
        self.num_channels = num_channels
        self._info = None
        self._memmap = np.memmap(
            filename,
            dtype=np.uint16,
            mode="r",
            shape=(
                self.fileinfo("frames"),
                self.fileinfo("height"),
                self.fileinfo("width"),
            ),
        )
        if self.fileinfo("endian") == 1:
            self._memmap = self._memmap.byteswap()

    def __getitem__(self, key):
        return self._memmap[key]

    def logmsg(self, message):
        return f"Dax file {self.filename} - {message}"

    def fileinfo(self, tag):
        if self._info is None:
            self.load_fileinfo()
        return self._info[tag]

    def load_fileinfo(self):
        with open(self.filename.replace(".dax", ".inf")) as f:
            infodata = f.read()
        self._info = {}
        m = re.search(r"frame dimensions = ([\d]+) x ([\d]+)", infodata)
        self._info["height"] = int(m.group(1))
        self._info["width"] = int(m.group(2))
        m = re.search(r"number of frames = ([\d]+)", infodata)
        self._info["frames"] = int(m.group(1))
        m = re.search(r" (big|little) endian", infodata)
        self._info["endian"] = 1 if m.group(1) == "big" else 0

    def channel(self, channel):
        return self._memmap[channel :: self.num_channels, :, :]

    def get_entire_image(self):
        return self._memmap[:, :, :]

    def zslice(self, zslice, channel=None):
        """If channel is None, return all channels."""
        if channel is None:
            return self._memmap[
                zslice * self.num_channels : zslice * self.num_channels
                + self.num_channels,
                :,
                :,
            ]
        else:
            return self._memmap[channel + zslice * self.num_channels, :, :]

    def frame(self, frame):
        return self._memmap[frame, :, :]

    def max_projection(self, channel=None):
        if channel is None:
            return np.max(self._memmap[:, :, :], axis=0)
        else:
            return np.max(self._memmap[channel :: self.num_channels, :, :], axis=0)

    def block_range(self, xr=None, yr=None, zr=None, channel=None):
        if xr is None:
            xr = (0, self.fileinfo("width"))
        if yr is None:
            yr = (0, self.fileinfo("height"))
        if zr is None:
            zr = (0, self.fileinfo("frames"))
        if xr[0] < 0:
            xr = (0, xr[1])
        if yr[0] < 0:
            yr = (0, yr[1])
        if zr[0] < 0:
            zr = (0, zr[1])
        if channel is None:
            zslice = slice(self.num_channels * zr[0], self.num_channels * (zr[1] + 1))
        else:
            zslice = slice(
                self.num_channels * zr[0] + channel,
                self.num_channels * (zr[1] + 1) + channel,
                self.num_channels,
            )
        yslice = slice(yr[0], yr[1] + 1)
        xslice = slice(xr[0], xr[1] + 1)
        return self._memmap[zslice, yslice, xslice], (zslice, yslice, xslice)

    def block(self, center, volume, channel=None):
        zr = (center[0] - volume[0], center[0] + volume[0])
        yr = (center[1] - volume[1], center[1] + volume[1])
        xr = (center[2] - volume[2], center[2] + volume[2])
        return self.block_range(zr=zr, yr=yr, xr=xr, channel=channel)
