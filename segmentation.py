import os
import glob
import pickle
import itertools
from functools import cached_property
from collections import defaultdict, Counter

import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import config


def get_slice(diff):
    overlap = int(config.get("mask_size") * diff / 220)
    if overlap == 0:
        return slice(None)
    elif overlap > 0:
        return slice(overlap, None)
    else:
        return slice(None, overlap)


def find_fov_overlaps(positions):
    fovsize = 220
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
            overlaps.append(
                [
                    (i, get_slice(diff[0]), get_slice(-diff[1])),
                    (fov, get_slice(-diff[0]), get_slice(diff[1])),
                ]
            )
    return overlaps


def get_slice_range(sliceobj):
    start = sliceobj.start if sliceobj.start is not None else 0
    stop = sliceobj.stop if sliceobj.stop is not None else config.get("mask_size")
    return range(start, stop)


def get_coords_in_slice(xslice, yslice):
    return ((x, y) for x in get_slice_range(xslice) for y in get_slice_range(yslice))


# This should probably be in the MaskList class
def get_overcounts(overlaps, masks, celldata):
    # We need to get a set of all mask pixels in the overlaps, and make sure we don't include pixels
    # more than once if they are in areas included in multiple overlaps (like at the corners)
    coords = defaultdict(set)
    for a, b in tqdm(overlaps, desc="Getting overlapping pixels"):
        coords[a[0]].update(get_coords_in_slice(a[1], a[2]))
        coords[b[0]].update(get_coords_in_slice(b[1], b[2]))

    cells = set(celldata[celldata["cell_id"].duplicated()]["cell_id"])
    dfs = []
    for fov, pointset in tqdm(
        coords.items(), desc="Getting cell volumes inside overlaps"
    ):
        times = defaultdict(list)
        for point, occurrences in Counter(pointset).items():
            times[occurrences].append(point)
        volumes = None
        for occurrences, points in times.items():
            xinds = [p[0] for p in points]
            yinds = [p[1] for p in points]
            if len(masks[fov].shape) == 2:
                unique, counts = np.unique(masks[fov][xinds, yinds], return_counts=True)
            elif len(masks[fov].shape) == 3:
                unique, counts = np.unique(
                    masks[fov][:, xinds, yinds], return_counts=True
                )
            df = pd.DataFrame(
                zip(itertools.repeat(fov), unique, counts),
                columns=["fov", "cell_id", "volume"],
            )
            df = df[df["cell_id"].isin(cells)].set_index("cell_id")
            df["volume"] /= occurrences + 1
            if volumes is None:
                volumes = df
            else:
                volumes["volume"] = volumes["volume"].add(df["volume"], fill_value=0)
        if volumes is not None:
            dfs.append(volumes.reset_index())
    return pd.concat(dfs)


def add_total_volume(celldata, overcounts):
    res = (
        celldata.groupby("cell_id")["volume"]
        .sum()
        .subtract(overcounts.groupby("cell_id")["volume"].sum(), fill_value=0)
    )
    return pd.merge(
        celldata, res, left_on="cell_id", right_on="cell_id", suffixes=("_fov", "")
    )


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


def get_global_cell_positions(celldata, positions, masksize):
    gxs = []
    gys = []
    for _, cell in celldata.iterrows():
        gxs.append(
            220 * (masksize - cell["fov_y"]) / masksize
            + positions.loc[int(cell["fov"])]["x"]
        )
        gys.append(
            220 * cell["fov_x"] / masksize + positions.loc[int(cell["fov"])]["y"]
        )
    gxs = [-x for x in gxs]
    celldata["global_x"] = gxs
    celldata["global_y"] = gys
    global_celldata = celldata.groupby("cell_id")[["global_x", "global_y"]].mean()
    return global_celldata


def save_cell_links(links, filename):
    with open(filename, "w") as f:
        for link in links:
            print(",".join([str(cell) for cell in link]), file=f)


def load_cell_links(filename):
    links = []
    with open(filename) as f:
        for line in f:
            links.append(set(int(cell) for cell in line.split(",")))
    return links


class MaskList:
    def __init__(self, mfx, segmask_dir=None, files=None, masks=None):
        self.files = files
        self.mfx = mfx
        self._masks = masks
        self._fov_renamed = False
        self._link_renamed = False
        if files is None:
            if glob.glob(os.path.join(segmask_dir, "Conv_zscan_*.png")):
                filename = "Conv_zscan_H0_F_{fov:03d}_cp_masks.png"
            elif glob.glob(os.path.join(segmask_dir, "Fov-*_seg.pkl")):
                filename = "Fov-{fov:04d}_seg.pkl"
            elif glob.glob(os.path.join(segmask_dir, "Conv_zscan_*.npy")):
                filename = "Conv_zscan_H0_F_{fov:03d}.npy"
            self.files = {
                fov: os.path.join(segmask_dir, filename.format(fov=fov))
                for fov in mfx.fovs
            }
        if masks is None:
            self._masks = {}

    def __getitem__(self, fov):
        if fov not in self._masks:
            if self.files[fov].endswith(".png"):
                self._masks[fov] = np.asarray(PIL.Image.open(self.files[fov]))
            elif self.files[fov].endswith(".pkl"):
                pkl = pickle.load(open(self.files[fov], "rb"))
                self._masks[fov] = pkl[0].astype(np.uint32)
            elif self.files[fov].endswith(".npy"):
                self._masks[fov] = np.load(self.files[fov]).astype(np.uint32)
        return self._masks[fov]

    def __len__(self):
        return len(self._masks)

    @property
    def dimensions(self):
        return len(self[0].shape)

    @cached_property
    def overlaps(self):
        return find_fov_overlaps(self.mfx.positions)

    @cached_property
    def links(self):
        cell_link_file = config.path("cell_links.csv")
        if os.path.exists(cell_link_file) and not config.get("rerun"):
            print("Loading existing cell links")
            links = load_cell_links(cell_link_file)
        else:
            links = self.match_cells_in_overlaps()
            save_cell_links(links, cell_link_file)
        return links

    def rename_cells_with_fov(self):
        if self._fov_renamed:
            return
        for fov in tqdm(self.mfx.fovs, desc="Adding FOV to cell names"):
            mask = self[fov].copy()
            mask[mask > 0] += fov * 10000
            self._masks[fov] = mask
        self._fov_renamed = True

    def link_cells_in_overlaps(self):
        self.rename_cells_with_fov()
        self.rename_linked_cells()

    def match_cells_in_overlaps(self):
        pairs = set()
        for a, b in tqdm(self.overlaps, desc="Linking cells in overlaps"):
            mask_a = self[a[0]]
            mask_b = self[b[0]]
            if self.dimensions == 2:
                strip_a = mask_a[a[1], a[2]]
                strip_b = mask_b[b[1], b[2]]
            elif self.dimensions == 3:
                strip_a = mask_a[:, a[1], a[2]]
                strip_b = mask_b[:, b[1], b[2]]
            df = pd.DataFrame(
                [
                    x
                    for x in list(zip(strip_a.flatten(), strip_b.flatten()))
                    if x[0] > 0 and x[1] > 0
                ]
            )
            if df.empty:
                continue
            ctab = pd.crosstab(df[0], df[1])
            way1 = set(zip(ctab.idxmax(axis=1), ctab.idxmax(axis=1).index))
            way2 = set(ctab.idxmax(axis=0).items())
            pairs.update(way1 & way2)
        linked_sets = [set([a, b]) for a, b in pairs]
        changed = True
        while changed:
            changed = False
            new = []
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

    def rename_linked_cells(self):
        if self._link_renamed:
            return
        for links in tqdm(self.links, desc="Renaming linked cells"):
            cellids = list(links)
            cellid = cellids[0]
            for other_id in cellids[1:]:
                fov = other_id // 10000
                self[fov][self[fov] == other_id] = cellid
        self._link_renamed = True

    def create_metadata_table(self, exclude_edge=False, use_overlaps=False):
        """Create a metadata table for cells."""
        if use_overlaps:
            self.link_cells_in_overlaps()
        dfs = []
        for fov in tqdm(self.mfx.fovs, desc="Creating cell metadata table"):
            mask = self[fov]
            if exclude_edge:
                # TODO: Only works for 2D masks
                # We don't exlude edges in the standard pipeline, so doesn't matter currently
                edge = np.unique(
                    np.concatenate([mask[:, 0], mask[:, -1], mask[0, :], mask[-1, :]])
                )
            else:
                edge = set()

            d = mask.ravel()
            f = lambda x: np.unravel_index(x.index, mask.shape)
            inds = pd.Series(d).groupby(d).apply(f).drop(0)
            pos = inds.apply(lambda x: np.array(x).mean(axis=1))
            df = pd.DataFrame(pos.values.tolist(), index=pos.index)
            if len(df) > 0:
                df.index.name = "cell_id"
                if self.dimensions == 2:
                    df.columns = ["fov_y", "fov_x"]
                elif self.dimensions == 3:
                    df.columns = ["z", "fov_y", "fov_x"]
                df["fov"] = fov
                counts = np.unique(mask, return_counts=True)
                df["fov_volume"] = pd.Series(counts[1], index=counts[0]).drop(0)
                # df["fov_volume"] *= 100  # Adjustment for downscaled mask
                dfs.append(df.reset_index())

        celldata = pd.concat(dfs, ignore_index=True)
        if use_overlaps:
            overcounts = get_overcounts(self.overlaps, self, celldata)
            res = (
                celldata.groupby("cell_id")["fov_volume"]
                .sum()
                .subtract(overcounts.groupby("cell_id")["volume"].sum(), fill_value=0)
            )
            celldata = pd.merge(
                celldata,
                pd.DataFrame(res, columns=["volume"]).astype(np.uint16),
                left_on="cell_id",
                right_index=True,
            )
        return celldata
