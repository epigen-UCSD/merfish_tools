"""Provides the CellSegmentation class for working with segmentation masks."""
import math
from functools import partial, cached_property
from collections import defaultdict, namedtuple
from typing import List, Set
from pathlib import Path

from cellpose import models as cpmodels
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import expand_labels

from . import config
from . import stats
from . import fileio
from . import util

Overlap = namedtuple("Overlap", ["fov", "xslice", "yslice"])


def get_slice(diff: float, fovsize: int = 220, get_trim: bool = False) -> slice:
    """Get a slice for the region of an image overlapped by another FOV.

    :param diff: The amount of overlap in the global coordinate system.
    :param fovsize: The width/length of a FOV in the global coordinate system, defaults to 220.
    :param get_trim: If True, return the half of the overlap closest to the edge. This is for
        determining in which region the barcodes should be trimmed to avoid duplicates.
    :return: A slice in the FOV coordinate system for the overlap.
    """
    if int(diff) == 0:
        return slice(None)
    elif diff > 0:
        if get_trim:
            diff = fovsize - ((fovsize - diff) / 2)
        overlap = (2048 * diff / fovsize) / config.get("scale")
        return slice(math.trunc(overlap), None)
    else:
        if get_trim:
            diff = -fovsize - ((-fovsize - diff) / 2)
        overlap = (2048 * diff / fovsize) / config.get("scale")
        return slice(None, math.trunc(overlap))


def find_fov_overlaps(
    positions: pd.DataFrame, fovsize: int = 220, get_trim: bool = False
) -> List[list]:
    """Identify overlaps between FOVs."""
    nn = NearestNeighbors()
    nn = nn.fit(positions)
    res = nn.radius_neighbors(
        positions, radius=fovsize, return_distance=True, sort_results=True
    )
    overlaps = []
    pairs = set()
    for i, (dists, fovs) in enumerate(zip(*res)):
        i = positions.iloc[i].name
        for dist, fov in zip(dists, fovs):
            fov = positions.iloc[fov].name
            if dist == 0 or (i, fov) in pairs:
                continue
            pairs.update([(i, fov), (fov, i)])
            diff = positions.loc[i] - positions.loc[fov]
            _get_slice = partial(get_slice, fovsize=fovsize, get_trim=get_trim)
            overlaps.append(
                [
                    Overlap(i, _get_slice(diff[0]), _get_slice(-diff[1])),
                    Overlap(fov, _get_slice(-diff[0]), _get_slice(diff[1])),
                ]
            )
    return overlaps


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
    print(
        f"Tagged {len(celldata[celldata['status'] == 'Too small'])} cells with volume < {min_volume} pixels"
    )

    # Remove large cells
    median = np.median(celldata[celldata["status"] != "Too small"]["volume"])
    celldata.loc[celldata["volume"] > median * max_factor, "status"] = "Too big"
    print(
        f"Tagged {len(celldata[celldata['status'] == 'Too big'])} cells with volume > {median*max_factor} pixels"
    )

    return celldata


class CellSegmentation:
    """A collection of segmentation masks from all FOVs."""

    def __init__(
        self,
        mask_folder: str,
        output: fileio.MerfishAnalysis = None,
        positions: pd.DataFrame = None,
        images: fileio.ImageDataset = None,
    ) -> None:
        """Initialize the instance.

        :param folderpath: The path to the folder containing the masks.
        :param output: The `MerfishAnalysis` object for saving and loading results.
        :param positions: The positions table representing the global coordinates of
            each FOV. See `fileio.MerlinOutput` for loading this file.
        """
        self.path = Path(mask_folder)
        self.output = output
        self.positions = positions
        self.images = images
        if images is not None:
            self.model = cpmodels.Cellpose(gpu=True, model_type="cyto2")
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
            except FileNotFoundError:
                segim, mask, flow, diams = self.segment_fov(key)
                filename = self.path / self.images.filename(0, key).parts[-1]
                fileio.save_mask(filename, (segim, mask, flow, diams))
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
            table["global_x"], table["global_y"] = util.fov_to_global_coordinates(
                table["fov_x"], table["fov_y"], table["fov"], self.positions
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

    @cached_property
    def fov_overlaps(self):
        return find_fov_overlaps(self.positions)

    def find_overlapping_cells(self) -> List[set]:
        """Identify the cells overlapping FOVs that are the same cell."""
        pairs = set()
        for a, b in tqdm(self.fov_overlaps, desc="Linking cells in overlaps"):
            # Get portions of masks that overlap
            if len(self[a.fov].shape) == 2:
                strip_a = self[a.fov][a.xslice, a.yslice]
                strip_b = self[b.fov][b.xslice, b.yslice]
            elif len(self[a.fov].shape) == 3:
                strip_a = self[a.fov][:, a.xslice, a.yslice]
                strip_b = self[b.fov][:, b.xslice, b.yslice]
            newpairs = match_cells_in_overlap(strip_a, strip_b)
            pairs.update(
                {(a.fov * 10000 + x[0], b.fov * 10000 + x[1]) for x in newpairs}
            )
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
        segim = self.images.load_image(fov=fov, channel="segmentation")
        mask, flow, _, diams = self.model.eval(
            segim,
            channels=[0, 0],
            diameter=80,
            cellprob_threshold=-4,
            flow_threshold=1.25,
        )
        for cellid, size in zip(*np.unique(mask, return_counts=True)):
            if size < 2500:
                mask[mask == cellid] = 0
        mask = expand_labels(mask, 3)
        return segim, mask, flow, diams

    def make_metadata_table(self) -> pd.DataFrame:
        def get_centers(inds):
            return np.mean(np.unravel_index(inds, shape=self[0].shape), axis=1)

        rows = []
        for fov, mask in tqdm(enumerate(self), desc="Getting cell volumes and centers"):
            # Some numpy tricks here. Confusing but fast.
            flat = mask.flatten()
            cells, split_inds, volumes = np.unique(
                np.sort(flat), return_index=True, return_counts=True
            )
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
        for a, b in self.fov_overlaps:
            fov_overlaps[a.fov].append(a)
            fov_overlaps[b.fov].append(b)
        cells = []
        volumes = []
        for fov, fov_over in tqdm(
            fov_overlaps.items(), desc="Calculating overlap volumes"
        ):
            for overlap in fov_over:
                counts = np.unique(
                    self[fov][overlap.xslice, overlap.yslice], return_counts=True
                )
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
