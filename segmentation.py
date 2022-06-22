import math
from functools import partial
from collections import defaultdict, namedtuple
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import config
import stats

Overlap = namedtuple("Overlap", ["fov", "xslice", "yslice"])


def get_slice(diff, fovsize=220, get_trim=False):
    if diff == 0:
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


def find_overlapping_cells(
    overlaps: List[list], masks: Dict[int, np.ndarray]
) -> List[set]:
    """Find cells in overlapping FOVs that are the same cell."""
    pairs = set()
    for a, b in tqdm(overlaps, desc="Linking cells in overlaps"):
        # Get portions of masks that overlap
        if len(masks[a.fov].shape) == 2:
            strip_a = masks[a.fov][a.xslice, a.yslice]
            strip_b = masks[b.fov][b.xslice, b.yslice]
        elif len(masks[a.fov].shape) == 3:
            strip_a = masks[a.fov][:, a.xslice, a.yslice]
            strip_b = masks[b.fov][:, b.xslice, b.yslice]
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


def make_metadata_table(masks):
    def get_centers(inds):
        return np.mean(np.unravel_index(inds, shape=masks[0].shape), axis=1)

    rows = []
    for fov, mask in tqdm(
        list(enumerate(masks)), desc="Getting cell volumes and centers"
    ):
        flat = mask.flatten()
        cells, split_inds, volumes = np.unique(
            np.sort(flat), return_index=True, return_counts=True
        )
        cell_inds = np.split(flat.argsort(), split_inds)[2:]
        centers = list(map(get_centers, cell_inds))
        if len(centers) > 0:
            coords = np.stack(centers, axis=0)
            df = pd.DataFrame([cells[1:], volumes[1:]] + coords.T.tolist()).T
            df["fov"] = fov
            rows.append(df)
    df = pd.concat(rows)
    if len(masks[0].shape) == 2:
        columns = ["fov_cell_id", "fov_volume", "fov_y", "fov_x", "fov"]
    elif len(masks[0].shape) == 3:
        columns = ["fov_cell_id", "fov_volume", "fov_z", "fov_y", "fov_x", "fov"]
    df.columns = columns
    df["cell_id"] = (df["fov"] * 10000 + df["fov_cell_id"]).astype(int)
    df["fov_volume"] = df["fov_volume"].astype(int)
    df = df.set_index("cell_id")
    df["fov_x"] *= config.get("scale")
    df["fov_y"] *= config.get("scale")
    stats.set("Segmented cells", len(df))
    stats.set(
        "Segmented cells per FOV", stats.get("Segmented cells") / stats.get("FOVs")
    )
    return df


def add_overlap_volume(celldata, overlaps, masks):
    fov_overlaps = defaultdict(list)
    for a, b in overlaps:
        fov_overlaps[a.fov].append(a)
        fov_overlaps[b.fov].append(b)
    cells = []
    volumes = []
    for fov, fov_over in tqdm(fov_overlaps.items(), desc="Calculating overlap volumes"):
        for overlap in fov_over:
            counts = np.unique(
                masks[fov][overlap.xslice, overlap.yslice], return_counts=True
            )
            cells.extend(counts[0] + fov * 10000)
            volumes.extend(counts[1])
    df = pd.DataFrame(np.array([cells, volumes]).T, columns=["cell", "volume"])
    celldata["overlap_volume"] = df.groupby("cell").max()


def add_linked_volume(celldata, cell_links):
    celldata["nonoverlap_volume"] = celldata["fov_volume"] - celldata["overlap_volume"]
    celldata["volume"] = np.nan
    for links in tqdm(cell_links, desc="Combining cell volumes in overlaps"):
        group = celldata[celldata.index.isin(links)]
        celldata.loc[celldata.index.isin(links), "volume"] = (
            group["overlap_volume"].mean() + group["nonoverlap_volume"].sum()
        )
    celldata.loc[celldata["volume"].isna(), "volume"] = celldata["fov_volume"]
    stats.set("Median cell volume (pixels)", np.median(celldata["volume"]))


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
