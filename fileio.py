"""Module for loading and saving files and data related to MERFISH experiments."""
import re
import json
import glob
import pickle
from pathlib import Path
from typing import Optional, Dict, Sequence

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_vizgen_barcodes(output_folder: str) -> pd.DataFrame:
    bcs = []
    for bcfile in glob.glob(f"{output_folder}/region_*/detected_transcripts.csv"):
        bcs.append(pd.read_csv(bcfile, index_col=0))
        region = bcfile.split("/")[-2].split("_")[1]
        bcs[-1]["region"] = region
    return pd.concat(bcs).reset_index().drop(columns="index")


def search_for_mask_file(segmask_dir: Path, fov: int) -> Path:
    """Find the filename for the segmentation mask of the given FOV."""
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
    for filename in segmask_dir.glob(f"*{fov}*"):
        for pattern in patterns:
            if re.search(pattern, str(filename)):
                return filename
    raise Exception(f"No mask found in {segmask_dir} for FOV {fov}")


def load_mask(segmask_dir: Path, fov: int) -> np.ndarray:
    """Load the segmentation mask for the given FOV."""
    filename = str(search_for_mask_file(segmask_dir, fov))

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            pkl = pickle.load(f)
        return pkl[0].astype(np.uint32)
    if filename.endswith(".npy"):
        return np.load(filename, allow_pickle=True).item()["masks"]
    if filename.endswith(".png"):
        from PIL import Image

        return np.asarray(Image.open(filename))

    raise Exception(f"Unknown format for mask {filename}")


def load_all_masks(segmask_dir: str, n_fovs: int):
    """Load all masks."""
    return [
        load_mask(Path(segmask_dir), fov)
        for fov in tqdm(range(n_fovs), desc="Loading cell masks")
    ]


def save_barcode_table(barcodes, filename) -> None:
    barcodes.to_csv(filename, index=False)


def load_barcode_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def save_cell_by_gene_table(cellbygene, filename) -> None:
    cellbygene.to_csv(filename)


def load_cell_by_gene_table(filename):
    return pd.read_csv(filename, index_col=0)


def save_stats(stats, filename) -> None:
    text = json.dumps(stats, indent=4)
    with open(filename, "w", encoding="utf8") as f:
        f.write(text)


def load_stats(filename):
    return json.load(open(filename, encoding="utf8"))


def _parse_list(inputString: str, dtype=float):
    if "," in inputString:
        return np.fromstring(inputString.strip("[] "), dtype=dtype, sep=",")
    else:
        return np.fromstring(inputString.strip("[] "), dtype=dtype, sep=" ")


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)


def load_data_organization(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        filename, converters={"frame": _parse_int_list, "zPos": _parse_list}
    )


class MerfishAnalysis:
    """A class for saving and loading results from this software package."""

    def __init__(self, folderpath: str) -> None:
        self.root = Path(folderpath)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_cell_metadata(self, celldata: pd.DataFrame) -> None:
        celldata.to_csv(self.root / "cell_metadata.csv")

    def load_cell_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "cell_metadata.csv", index_col=0)

    def save_linked_cells(self, links) -> None:
        with open(self.root / "linked_cells.txt", "w", encoding="utf8") as f:
            for link in links:
                print(repr(link), file=f)

    def load_linked_cells(self):
        links = []
        with open(self.root / "linked_cells.txt", encoding="utf8") as f:
            for line in f:
                links.append(eval(line))
        return links


class MerlinOutput:
    """A class for loading results from a MERlin output folder."""

    def __init__(self, folderpath: str) -> None:
        self.root = Path(folderpath)

    def n_fovs(self) -> int:
        """Return the number of FOVs in the experiment."""
        path = self.root / "Decode" / "barcodes"
        return len(list(path.glob("barcode_data_*.h5")))

    def load_raw_barcodes(self, fov: int) -> pd.DataFrame:
        """Load detailed barcode metadata from the Decode folder."""
        path = self.root / "Decode" / "barcodes" / f"barcode_data_{fov}.h5"
        return pd.read_hdf(path)

    def count_raw_barcodes(self, fov: int) -> int:
        """Count the number of barcodes for an fov in the Decode folder."""
        path = self.root / "Decode" / "barcodes" / f"barcode_data_{fov}.h5"
        barcodes = h5py.File(path, "r")
        return len(barcodes["barcodes/table"])

    def load_filtered_barcodes(self, fov: int) -> pd.DataFrame:
        """Load detailed barcode metadata from the AdaptiveFilterBarcodes folder."""
        path = (
            self.root / "AdaptiveFilterBarcodes" / "barcodes" / f"barcode_data_{fov}.h5"
        )
        return pd.read_hdf(path)

    def load_exported_barcodes(self) -> pd.DataFrame:
        """Load the exported barcode table."""
        path = self.root / "ExportBarcodes" / "barcodes.csv"
        return pd.read_csv(path)

    def load_drift_transformations(self, fov: int):
        """Get the drifts calculated between hybridization rounds for the given FOV.

        Returns a numpy array containing scikit-image SimilarityTransform objects.
        The load_hyb_drifts function can be used instead to convert this to a
        pandas DataFrame.
        """
        possible_folders = ["FiducialBeadWarp", "FiducialCorrelationWarp"]
        for folder in possible_folders:
            if Path(self.root, folder).exists():
                path = self.root / folder / "transformations" / f"offsets_{fov}.npy"
                return np.load(path, allow_pickle=True)
        raise Exception("Could not find drift transformations")

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

    def load_data_organization(self) -> pd.DataFrame:
        """Load the data organization table."""
        path = self.root / "dataorganization.csv"
        return load_data_organization(path)


class ImageDataset:
    def __init__(self, folderpath: str, data_organization: str = None) -> None:
        self.root = Path(folderpath)
        self.filenames = list(self.root.glob("*.dax"))
        if isinstance(data_organization, str):
            self.data_organization = load_data_organization(data_organization)
        elif isinstance(data_organization, pd.DataFrame):
            self.data_organization = data_organization
        if self.data_organization is not None:
            self.regex = re.compile(self.data_organization.iloc[0]["imageRegExp"])

    def filename(self, hyb, fov) -> Path:
        """Locates the filename for the image of the given hyb round and FOV."""
        for file in self.filenames:
            match = self.regex.search(str(file))
            if match is not None:
                props = match.groupdict()
                if int(props["imagingRound"]) == hyb and int(props["fov"]) == fov:
                    return file
        return None

    def load_image(
        self,
        fov: int,
        zslice: int = None,
        bit: int = None,
        hyb: int = None,
        channel: str = None,
        max_projection: bool = False,
    ) -> np.ndarray:
        """Load an image from the dataset.

        The image to load can be specified by passing either the bit or the
        hybridization round and color channel. If the zslice to be loaded is
        not specified, then either a 3D image containing all z-slices, or
        a 2D max projection along the z-axis is returned, depending on the
        max_projection parameter.
        """
        if hyb is not None:
            filename = self.filename(hyb, fov)
            if isinstance(channel, str):
                frames = self.data_organization[
                    (self.data_organization["color"] == channel)
                    & (self.data_organization["imagingRound"] == hyb)
                ].iloc[0]["frame"]
        elif bit is not None:
            bitrow = self.data_organization[
                self.data_organization["bitNumber"] == bit
            ].iloc[0]
            filename = self.filename(bitrow["imagingRound"], fov)
            frames = bitrow["frame"]
        else:
            raise Exception("Must specify hyb or bit")

        dax = DaxFile(str(filename))
        if zslice is not None:
            return dax.frame(frames[zslice])
        imgstack = np.array([dax.frame(frame) for frame in frames])
        if max_projection:
            return imgstack.max(axis=0)
        return imgstack


class DaxFile:
    """Loads data from a DAX image file."""

    def __init__(self, filename: str, num_channels: Optional[int] = None) -> None:
        """Note: If num_channels=None, the channel and zslice functions will not work."""
        self.filename = filename
        self.num_channels = num_channels
        self._info: Dict[str, int] = {}
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

    def __getitem__(self, key: slice) -> np.ndarray:
        return self._memmap[key]

    def fileinfo(self, tag: str) -> int:
        """Get a property from the .inf associated file."""
        if not self._info:
            self.load_fileinfo()
        return self._info[tag]

    def load_fileinfo(self) -> None:
        """Load the .inf associated file and parse the info."""
        with open(self.filename.replace(".dax", ".inf"), encoding="utf8") as f:
            infodata = f.read()
        self._info = {}
        m = re.search(r"frame dimensions = ([\d]+) x ([\d]+)", infodata)
        self._info["height"] = int(m.group(1))
        self._info["width"] = int(m.group(2))
        m = re.search(r"number of frames = ([\d]+)", infodata)
        self._info["frames"] = int(m.group(1))
        m = re.search(r" (big|little) endian", infodata)
        self._info["endian"] = 1 if m.group(1) == "big" else 0

    def channel(self, channel: int) -> np.ndarray:
        """Return a 3D array of all z-slices of the given channel"""
        return self._memmap[channel :: self.num_channels, :, :]

    def get_entire_image(self) -> np.ndarray:
        return self._memmap[:, :, :]

    def zslice(self, zslice: int, channel: Optional[int] = None) -> np.ndarray:
        """Return a z-slice of the image.

        If channel is None, returns a 3D array with all channels, otherwise returns
        a 2D array with the given channel.
        """
        if self.num_channels is None:
            raise Exception("num_channels must be specified to use this function")
        if channel is None:
            return self._memmap[
                zslice * self.num_channels : zslice * self.num_channels
                + self.num_channels,
                :,
                :,
            ]
        return self._memmap[channel + zslice * self.num_channels, :, :]

    def frame(self, frame: int) -> np.ndarray:
        """Return a 2D array of the given frame."""
        return self._memmap[frame, :, :]

    def max_projection(self, channel: int) -> np.ndarray:
        """Return a max projection across z-slices of the given channel."""
        return np.max(self._memmap[channel :: self.num_channels, :, :], axis=0)

    def block_range(
        self,
        xr: Optional[Sequence[int]] = None,
        yr: Optional[Sequence[int]] = None,
        zr: Optional[Sequence[int]] = None,
        channel: Optional[int] = None,
    ):
        """Return a 3D block of the image specific by the given x, y, and z ranges."""
        if self.num_channels is None:
            raise Exception("num_channels must be specified to use this function")
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

    def block(
        self,
        center: Sequence[int],
        volume: Sequence[int],
        channel: Optional[int] = None,
    ):
        """Return a 3D block of the image specified by the given center and volume.

        The volume specifies the radius of the block in each dimension.
        """
        zr = (center[0] - volume[0], center[0] + volume[0])
        yr = (center[1] - volume[1], center[1] + volume[1])
        xr = (center[2] - volume[2], center[2] + volume[2])
        return self.block_range(zr=zr, yr=yr, xr=xr, channel=channel)
