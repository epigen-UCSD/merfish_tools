"""Module for loading and saving files and data related to MERFISH experiments."""
import re
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Sequence

import h5py
import numpy as np
import pandas as pd


def search_for_mask_file(segmask_dir: Path, fov: int) -> Path:
    """Find the filename for the segmentation mask of the given FOV.

    This function searches the given directory for a file matching a number of different
    possible patterns for segmentation mask filenames, based on cellpose output naming and
    various other scripts used for segmentation internally at the USCD Center for Epigenomics.

    Args:
        segmask_dir: The directory containing segmentation masks.
        fov: The field of view to find the mask file for.

    Returns:
        The mask file found for the specified field of view.

    Raises:
        FileNotFoundError: If no mask file could be found.
    """
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
    raise FileNotFoundError(f"No mask found in {segmask_dir} for FOV {fov}")


def load_mask(segmask_dir: Path, fov: int) -> np.ndarray:
    """Load the segmentation mask for the given FOV.

    Args:
        segmask_dir: The directory containing segmentation masks.
        fov: Which field of view to the load the mask for.

    Returns:
        The segmentation mask for the specified field of view.
    """
    filename = str(search_for_mask_file(segmask_dir, fov))

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            pkl = pickle.load(f)
        return pkl[0].astype(np.uint32)
    if filename.endswith(".npy"):
        try:
            return np.load(filename, allow_pickle=True).item()["masks"]
        except ValueError:
            return np.load(filename)
    if filename.endswith(".png"):
        from PIL import Image

        return np.asarray(Image.open(filename))

    raise Exception(f"Unknown format for mask {filename}")


def save_mask(filename: Path, mask: np.ndarray) -> None:
    """Save a mask in cellpose format.

    Args:
        filename: A pathlib.Path for the save location.
        cellpose_data: A tuple containing the segmentation image and the masks, flows, and diams returned by cellpose.
    """
    filename.parents[0].mkdir(parents=True, exist_ok=True)
    # cpio.masks_flows_to_seg(*cellpose_data, filename, [0, 0])
    np.save(filename, mask)


def save_stats(stats, filename) -> None:
    text = json.dumps(stats, indent=4)
    with open(filename, "w", encoding="utf8") as f:
        f.write(text)


def load_stats(filename):
    return json.load(open(filename, encoding="utf8"))


def _parse_list(list_string: str, dtype=float):
    if "," in list_string:
        sep = ","
    else:
        sep = " "
    return np.fromstring(list_string.strip("[] "), dtype=dtype, sep=sep)


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)


def load_data_organization(filename: str) -> pd.DataFrame:
    """Load a data organization file into a pandas DataFrame.

    Args:
        filename: The path to the data organization file.

    Returns:
        A pandas DataFrame of the data organization.
    """
    return pd.read_csv(filename, converters={"frame": _parse_int_list, "zPos": _parse_list})


def load_fov_positions(path: Path) -> pd.DataFrame:
    """Get the global positions of the FOVs.

    The coordinates indicate the top-left corner of the FOV.

    Args:
        path: A pathlib.Path to the FOV positions file.

    Returns:
        A pandas DataFrame containing the FOV positions.
    """
    positions = pd.read_csv(path, header=None)
    positions.columns = ["x", "y"]
    return positions


class MerfishAnalysis:
    """A class for saving and loading results from this software package."""

    def __init__(self, folderpath: str, save_to_subfolder: str = "") -> None:
        self.root = Path(folderpath)
        self.save_path = self.root / save_to_subfolder
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __load_dataframe(self, name: str, add_region: bool) -> pd.DataFrame:
        filename = self.save_path / name
        if filename.exists():
            return pd.read_csv(filename, index_col=0)
        # Check if this is a multi-region MERSCOPE experiment
        if list(self.root.glob("region_*")):
            region_dfs = []
            for region in list(self.root.glob("region_*")):
                num = str(region).rsplit("_", maxsplit=1)[-1]
                dataframe = pd.read_csv(region / name, index_col=0)
                if add_region:
                    dataframe["region"] = num
                region_dfs.append(dataframe)
            dataframe = pd.concat(region_dfs)
            dataframe.to_csv(filename)
            return dataframe
        raise FileNotFoundError(filename)

    def save_cell_metadata(self, celldata: pd.DataFrame) -> None:
        celldata.to_csv(self.save_path / "cell_metadata.csv")

    def load_cell_metadata(self) -> pd.DataFrame:
        return self.__load_dataframe("cell_metadata.csv", add_region=True)

    def has_cell_metadata(self) -> bool:
        return Path(self.save_path, "cell_metadata.csv").exists()

    def save_linked_cells(self, links) -> None:
        with open(self.save_path / "linked_cells.txt", "w", encoding="utf8") as f:
            for link in links:
                print(repr(link), file=f)

    def load_linked_cells(self):
        links = []
        with open(self.save_path / "linked_cells.txt", encoding="utf8") as f:
            for line in f:
                links.append(eval(line))
        return links

    def save_barcode_table(self, barcodes, dask=False) -> None:
        if dask:
            barcodes.to_csv(self.save_path / "detected_transcripts")
        else:
            barcodes.to_csv(self.save_path / "detected_transcripts.csv")

    def load_barcode_table(self) -> pd.DataFrame:
        return self.__load_dataframe("detected_transcripts.csv", add_region=True)

    def save_cell_by_gene_table(self, cellbygene) -> None:
        cellbygene.to_csv(self.save_path / "cell_by_gene.csv")

    def load_cell_by_gene_table(self) -> pd.DataFrame:
        return self.__load_dataframe("cell_by_gene.csv", add_region=False)


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
        path = self.root / "AdaptiveFilterBarcodes" / "barcodes" / f"barcode_data_{fov}.h5"
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
        return load_fov_positions(path)

    def load_data_organization(self) -> pd.DataFrame:
        """Load the data organization table."""
        path = self.root / "dataorganization.csv"
        return load_data_organization(path)


class ImageDataset:
    def __init__(self, folderpath: str, data_organization: str = None) -> None:
        self.root = Path(folderpath)
        if isinstance(data_organization, str):
            self.data_organization = load_data_organization(data_organization)
        elif isinstance(data_organization, pd.DataFrame):
            self.data_organization = data_organization
        elif data_organization is None and Path(self.root, "dataorganization.csv").exists():
            self.data_organization = load_data_organization(self.root / "dataorganization.csv")
        if self.data_organization is not None:
            self.regex = {}
            for _, row in self.data_organization.iterrows():
                self.regex[row["channelName"]] = re.compile(row["imageRegExp"])
        if Path(self.root, "data").is_dir():
            self.filenames = list(Path(self.root, "data").glob("*.dax"))
        else:
            self.filenames = list(self.root.glob("*.dax"))

    def _find_filename(self, regex, image_type, fov, imaging_round=None) -> Path:
        """Locates the filename for the image of the given hyb round and FOV."""
        for file in self.filenames:
            match = regex.search(str(file.name))
            if match is not None:
                props = match.groupdict()
                if "imageType" in props and props["imageType"] != image_type:
                    continue
                if "fov" in props and int(props["fov"]) != fov:
                    continue
                if "imagingRound" in props and int(props["imagingRound"]) != imaging_round:
                    continue
                return file
        raise FileNotFoundError(f"Could not find image file for {image_type=}, {fov=}, {imaging_round=}")

    def filename(self, channel, fov) -> Path:
        row = self.data_organization[self.data_organization["channelName"] == channel].iloc[0]  # Assume 1 match
        return self._find_filename(self.regex[channel], row["imageType"], fov, row["imagingRound"])

    def n_fovs(self) -> int:
        hyb = str(self.filenames[0]).split("_")[-3]  # Get the hyb round of the first filename
        return sum(hyb in str(f) for f in self.filenames)  # Check how many filenames have that hyb

    def load_fov_positions(self):
        return load_fov_positions(self.root / "settings/positions.csv")

    def has_positions(self):
        return Path(self.root / "settings/positions.csv").exists()

    def load_image(
        self, fov: int, zslice: int = None, channel: str = None, max_projection: bool = False, fiducial: bool = False
    ) -> np.ndarray:
        """Load an image from the dataset.

        The image to load can be specified by passing either the bit or the
        hybridization round and color channel. If the zslice to be loaded is
        not specified, then either a 3D image containing all z-slices, or
        a 2D max projection along the z-axis is returned, depending on the
        max_projection parameter.
        """
        try:
            row = self.data_organization[self.data_organization["channelName"] == channel].iloc[0]  # Assume 1 match
        except IndexError:
            raise IndexError(f"Channel {channel} not found in data organization")
        filename = self.filename(channel, fov)
        dax = DaxFile(str(filename))
        if fiducial:
            return dax.frame(row["fiducialFrame"])
        if zslice is not None:
            return dax.frame(row["frame"][zslice])
        imgstack = np.array([dax.frame(frame) for frame in row["frame"]])
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
                zslice * self.num_channels : zslice * self.num_channels + self.num_channels,
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
