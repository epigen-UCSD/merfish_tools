"""Provides the CellSegmentation class for working with segmentation masks."""
from functools import cached_property
from collections import defaultdict
from typing import List, Set
from pathlib import Path

from cellpose import models as cpmodels, utils as cputils
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table

from . import config
from . import stats
from . import fileio
from . import images


def match_cells_in_overlap(strip_a: np.ndarray, strip_b: np.ndarray) -> Set[tuple]:
    """Find cells in overlapping regions of two FOVs that are the same cells.

    :param strip_a: The overlapping region of the segmentation mask from one FOV.
    :param strip_b: The overlapping region of the segmentation mask from another FOV.
    :return: A set of pairs of ints (tuples) representing the mask labels from each mask
        that are the same cell. For example, the tuple `(23, 45)` means mask label 23 from
        the mask given by `strip_a` is the same cell as mask label 45 in the mask given by
        `strip_b`.
    """
    # Pair up pixels in overlap regions
    # This could be more precise by drift correcting between the two FOVs
    p = np.array([strip_a.flatten(), strip_b.flatten()]).T
    # Remove pixel pairs with 0s (no cell) and count overlapping areas between cells
    ps, c = np.unique(p[np.all(p != 0, axis=1)], axis=0, return_counts=True)
    # For each cell from A, find the cell in B it overlaps with most (s1)
    # Do the same from B to A (s2)
    df = pd.DataFrame(np.hstack((ps, np.array([c]).T)), columns=["a", "b", "count"])
    s1 = {
        tuple(x)
        for x in df.sort_values(["a", "count"], ascending=[True, False])
        .groupby("a")
        .first()
        .reset_index()[["a", "b"]]
        .values.tolist()
    }
    s2 = {
        tuple(x)
        for x in df.sort_values(["b", "count"], ascending=[True, False])
        .groupby("b")
        .first()
        .reset_index()[["a", "b"]]
        .values.tolist()
    }
    # Only keep the pairs found in both directions
    return s1 & s2


def filter_by_volume(celldata, min_volume, max_factor):
    # Remove small cells
    celldata.loc[celldata["volume"] < min_volume, "status"] = "Too small"
    print(f"Tagged {len(celldata[celldata['status'] == 'Too small'])} cells with volume < {min_volume} pixels")

    # Remove large cells
    median = np.median(celldata[celldata["status"] != "Too small"]["volume"])
    celldata.loc[celldata["volume"] > median * max_factor, "status"] = "Too big"
    print(f"Tagged {len(celldata[celldata['status'] == 'Too big'])} cells with volume > {median*max_factor} pixels")

    return celldata


def remove_small_cells_from_mask(mask, min_volume):
    sizes = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
    mask[np.isin(mask, sizes[sizes["area"] < min_volume]["label"])] = 0
    return mask


class CellSegmentation:
    """A collection of segmentation masks from all FOVs."""

    def __init__(
        self,
        mask_folder: str = None,
        output: fileio.MerfishAnalysis = None,
        positions: pd.DataFrame = None,
        imagedata: fileio.ImageDataset = None,
        channel: str = "PolyT",
        zslice: int = None,
    ) -> None:
        """Initialize the instance.

        :param folderpath: The path to the folder containing the masks.
        :param output: The `MerfishAnalysis` object for saving and loading results.
        :param positions: The positions table representing the global coordinates of
            each FOV. See `fileio.MerlinOutput` for loading this file.
        """
        self.path = None
        if mask_folder:
            self.path = Path(mask_folder)
        self.output = output
        self.positions = None
        if positions is not None:
            self.positions = images.FOVPositions(positions=positions)
        self.imagedata = imagedata
        self.channel = channel
        self.zslice = zslice
        if imagedata is not None:
            self.model = cpmodels.Cellpose(gpu=True, model_type="cyto2")
            if not self.positions and imagedata.has_positions():
                self.positions = images.FOVPositions(positions=imagedata.load_fov_positions())
        self.masks = {}

    def __getitem__(self, key: int) -> np.ndarray:
        """Return the mask for the given FOV.

        The mask will be loaded into memory the first time it is requested, then
        stored for future requests. If the mask does not exist and an ImageDataset
        was given at construction, the segmentation mask will be created and saved.

        :param key: The FOV to return the mask for.
        :return: The segmentation mask.
        """
        if not hasattr(self, "masks"):
            self.masks = {}
        if key not in self.masks:
            try:
                self.masks[key] = fileio.load_mask(self.path, key)
            except (FileNotFoundError, AttributeError):
                mask = self.segment_fov(key)
                if self.path:
                    filename = self.path / self.imagedata.filename(self.channel, key).stem
                    fileio.save_mask(Path(str(filename) + "_seg.npy"), mask)
                self.masks[key] = mask
                return mask
        return self.masks[key]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self) -> np.ndarray:
        try:
            i = self.i
            self.i += 1
            return self[i]
        except Exception as exc:
            raise StopIteration from exc

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Get the cell metadata table.

        When the metadata table is accessed for the first time, it will first attempt
        to load a saved metadata table if `output` was given. If the file doesn't exist,
        the metadata table will be created and stored in memory. If `output` was given,
        the table will be saved to disk so it can be loaded in the future.

        :return: The cell metadata table.
        """
        # Try to load existing table
        if self.output is not None:
            try:
                return self.output.load_cell_metadata()
            except FileNotFoundError:
                pass  # Need to create it

        table = self.make_metadata_table()
        if self.positions is not None:
            table["global_x"], table["global_y"] = self.positions.local_to_global_coordinates(
                table["fov_x"], table["fov_y"], table["fov"]
            )
            table["overlap_volume"] = self.get_overlap_volume()
            self.__add_linked_volume(table)
            table = table.drop(
                [
                    "fov_cell_id",
                    "fov_volume",
                    "overlap_volume",
                    "nonoverlap_volume",
                ],
                axis=1,
            )
            duplicates = np.concatenate([list(group)[1:] for group in self.linked_cells])
            table = table.drop(duplicates, axis=0)
        if self.output is not None:
            self.output.save_cell_metadata(table)
        return table

    @cached_property
    def linked_cells(self) -> List[set]:
        if self.output is not None:
            try:
                return self.output.load_linked_cells()
            except FileNotFoundError:
                pass  # Need to create it
        cell_links = self.find_overlapping_cells()
        if self.output is not None:
            self.output.save_linked_cells(cell_links)
        return cell_links

    def find_overlapping_cells(self) -> List[set]:
        """Identify the cells overlapping FOVs that are the same cell."""
        pairs = set()
        for a, b in tqdm(self.positions.overlaps, desc="Linking cells in overlaps"):
            # Get portions of masks that overlap
            if len(self[a.fov].shape) == 2:
                strip_a = self[a.fov][a.xslice, a.yslice]
                strip_b = self[b.fov][b.xslice, b.yslice]
            elif len(self[a.fov].shape) == 3:
                strip_a = self[a.fov][:, a.xslice, a.yslice]
                strip_b = self[b.fov][:, b.xslice, b.yslice]
            newpairs = match_cells_in_overlap(strip_a, strip_b)
            pairs.update({(a.fov * 10000 + x[0], b.fov * 10000 + x[1]) for x in newpairs})
        linked_sets = [set([a, b]) for a, b in pairs]
        # Combine sets until they are all disjoint
        # e.g., if there is a (1, 2) and (2, 3) set, combine to (1, 2, 3)
        # This is needed for corners where 4 FOVs overlap
        changed = True
        while changed:
            changed = False
            new: List[set] = []
            for a in linked_sets:
                for b in new:
                    if not b.isdisjoint(a):
                        b.update(a)
                        changed = True
                        break
                else:
                    new.append(a)
            linked_sets = new
        return linked_sets

    def segment_fov(self, fov: int):
        segim = self.imagedata.load_image(fov=fov, channel=self.channel, zslice=self.zslice)
        if segim.ndim == 2:
            mask, _, _, _ = self.model.eval(
                segim,
                channels=[0, 0],
                diameter=80,
                cellprob_threshold=-4,
                flow_threshold=1.25,
            )
            mask = remove_small_cells_from_mask(mask, min_volume=2500)
            mask = expand_labels(mask, 3)
        elif segim.ndim == 3:
            frames, _, _, _ = self.model.eval(
                list(segim), diameter=80, channels=[0, 0], cellprob_threshold=-4, flow_threshold=1.25
            )
            mask = np.array(cputils.stitch3D(frames))
            mask = remove_small_cells_from_mask(mask, min_volume=2500)
            for i, frame in enumerate(mask):
                mask[i] = expand_labels(frame, 3)
        return mask

    def make_metadata_table(self) -> pd.DataFrame:
        def get_centers(inds):
            return np.mean(np.unravel_index(inds, shape=self[0].shape), axis=1)

        rows = []
        for fov, mask in tqdm(enumerate(self), desc="Getting cell volumes and centers"):
            # Some numpy tricks here. Confusing but fast.
            flat = mask.flatten()
            cells, split_inds, volumes = np.unique(np.sort(flat), return_index=True, return_counts=True)
            cell_inds = np.split(flat.argsort(), split_inds)[2:]
            centers = list(map(get_centers, cell_inds))
            if len(centers) > 0:
                coords = np.stack(centers, axis=0)
                row = pd.DataFrame([cells[1:], volumes[1:]] + coords.T.tolist()).T
                row["fov"] = fov
                rows.append(row)
        table = pd.concat(rows)
        if len(self[0].shape) == 2:
            columns = ["fov_cell_id", "fov_volume", "fov_y", "fov_x", "fov"]
        elif len(self[0].shape) == 3:
            columns = ["fov_cell_id", "fov_volume", "fov_z", "fov_y", "fov_x", "fov"]
        table.columns = columns
        table["cell_id"] = (table["fov"] * 10000 + table["fov_cell_id"]).astype(int)
        table["fov_cell_id"] = table["fov_cell_id"].astype(int)
        table["fov_volume"] = table["fov_volume"].astype(int)
        table = table.set_index("cell_id")
        table["fov_x"] *= config.get("scale")
        table["fov_y"] *= config.get("scale")
        stats.set("Segmented cells", len(table))
        try:
            stats.set(
                "Segmented cells per FOV",
                stats.get("Segmented cells") / stats.get("FOVs"),
            )
        except KeyError:
            pass
        return table

    def get_overlap_volume(self) -> None:
        fov_overlaps = defaultdict(list)
        for a, b in self.positions.overlaps:
            fov_overlaps[a.fov].append(a)
            fov_overlaps[b.fov].append(b)
        cells = []
        volumes = []
        for fov, fov_over in tqdm(fov_overlaps.items(), desc="Calculating overlap volumes"):
            for overlap in fov_over:
                counts = np.unique(self[fov][overlap.xslice, overlap.yslice], return_counts=True)
                cells.extend(counts[0] + fov * 10000)
                volumes.extend(counts[1])
        df = pd.DataFrame(np.array([cells, volumes]).T, columns=["cell", "volume"])
        return df.groupby("cell").max()

    def __add_linked_volume(self, table) -> None:
        table["nonoverlap_volume"] = table["fov_volume"] - table["overlap_volume"]
        table["volume"] = np.nan
        for links in tqdm(self.linked_cells, desc="Combining cell volumes in overlaps"):
            group = table[table.index.isin(links)]
            table.loc[table.index.isin(links), "volume"] = (
                group["overlap_volume"].mean() + group["nonoverlap_volume"].sum()
            )
        table.loc[table["volume"].isna(), "volume"] = table["fov_volume"]
        stats.set("Median cell volume (pixels)", np.median(table["volume"]))
