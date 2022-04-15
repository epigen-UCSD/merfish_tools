from functools import cached_property
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import norm

import fileio
import config


def process_merlin_barcodes(
    barcodes: pd.DataFrame,
    neighbors: faiss.IndexFlatL2,
    expanded_codebook: pd.DataFrame,
) -> pd.DataFrame:
    """Process the barcodes for a single field of view.

    The error type and bit are determined using an expanded codebook and returned as a
    DataFrame with extraneous columns removed.
    """
    X = np.ascontiguousarray(
        barcodes.filter(like="intensity_").to_numpy(), dtype=np.float32
    )
    indexes = neighbors.search(X, k=1)

    res = expanded_codebook.iloc[indexes[1].flatten()].copy()
    res = res.set_index(["name", "id"]).filter(like="bit").sum(axis=1)
    res = pd.DataFrame(res, columns=["bits"]).reset_index()

    df = barcodes[["barcode_id", "fov", "x", "y", "z"]]
    df = df.reset_index(drop=True)
    df["gene"] = res["name"]
    df["error_type"] = res["bits"] - 4
    df["error_bit"] = res["id"].str.split("flip", expand=True)[1].fillna(0)
    return df


def expand_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    """Add codes for every possible bit flip to a codebook."""
    books = [codebook]
    bits = len(codebook.filter(like="bit").columns)
    for bit in range(1, bits + 1):
        flip = codebook.copy()
        flip[f"bit{bit}"] = (~flip[f"bit{bit}"].astype(bool)).astype(int)
        flip["id"] = flip["id"] + f"_flip{bit}"
        books.append(flip)
    return pd.concat(books)


def normalize_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    """L2 normalize a codebook."""
    codes = codebook.filter(like="bit")
    normcodes = codes.apply(lambda row: row / norm(row), axis=1)
    return normcodes


def make_table(merlin_dir: str, codebook: pd.DataFrame) -> pd.DataFrame:
    """Create a table of all barcodes with error correction information."""
    codebook = expand_codebook(codebook)
    X = np.ascontiguousarray(normalize_codebook(codebook).to_numpy(), dtype=np.float32)
    neighbors = faiss.IndexFlatL2(X.shape[1])
    neighbors.add(X)
    dfs = []
    for barcodes in tqdm(fileio.merlin_barcodes(merlin_dir), desc="Preparing barcodes"):
        dfs.append(process_merlin_barcodes(barcodes, neighbors, codebook))
    df = pd.concat(dfs, ignore_index=True)
    df["status"] = "unprocessed"
    return df


def calculate_global_coordinates(
    barcodes: pd.DataFrame, positions: pd.DataFrame
) -> None:
    """Add global_x and global_y columns to barcodes."""

    def convert_to_global(group: pd.DataFrame) -> pd.DataFrame:
        """Calculate the global coordinates for a single FOV."""
        fov = int(group.iloc[0]["fov"])
        ypos = positions.loc[fov][0]
        xpos = positions.loc[fov][1]
        group["global_y"] = 220 * (2048 - group["y"]) / 2048 + ypos
        group["global_x"] = 220 * group["x"] / 2048 + xpos
        return group[["global_x", "global_y"]]

    barcodes[["global_x", "global_y"]] = barcodes.groupby("fov").apply(
        convert_to_global
    )


def assign_to_cells(barcodes, masks, drifts=None):
    cellids = []
    for fov, group in tqdm(barcodes.groupby("fov"), desc="Assigning barcodes to cells"):
        if drifts is not None:
            xdrift = drifts.loc[fov]["X drift"]
            ydrift = drifts.loc[fov]["Y drift"]
        else:
            xdrift, ydrift = 0, 0
        x = (
            round(group["x"] + xdrift)
            .astype(int)
            .apply(lambda n: 0 if n < 0 else 2047 if n > 2047 else n)
        )
        y = (
            round(group["y"] + ydrift)
            .astype(int)
            .apply(lambda n: 0 if n < 0 else 2047 if n > 2047 else n)
        )
        cellids.append(pd.Series(masks[fov][y, x], index=group.index))
    barcodes["cell_id"] = pd.concat(cellids)
    barcodes.loc[barcodes["cell_id"] != 0, "cell_id"] = (
        barcodes["fov"].astype(int) * 10000 + barcodes["cell_id"]
    )
    barcodes.loc[
        (barcodes["cell_id"] == 0) & (barcodes["status"] == "unprocessed"), "status"
    ] = "no cell"
    barcodes.loc[
        (barcodes["cell_id"] != 0) & (barcodes["status"] == "unprocessed"), "status"
    ] = "good"


def link_cell_ids(barcodes, cell_links):
    link_map = {cell: list(group)[0] for group in cell_links for cell in group}
    barcodes["cell_id"] = barcodes["cell_id"].apply(
        lambda cid: link_map[cid] if cid in link_map else cid
    )


def mark_barcodes_on_margins(barcodes, left=0, right=0, top=0, bottom=0):
    barcodes.loc[barcodes["x"] < left, "status"] = "edge"
    barcodes.loc[barcodes["y"] < top, "status"] = "edge"
    barcodes.loc[barcodes["x"] > 2048 - right, "status"] = "edge"
    barcodes.loc[barcodes["y"] > 2048 - bottom, "status"] = "edge"


def mark_barcodes_in_overlaps(barcodes, trim_overlaps):
    xstarts = defaultdict(list)
    xstops = defaultdict(list)
    ystarts = defaultdict(list)
    ystops = defaultdict(list)
    for pair in trim_overlaps:
        for overlap in pair:
            if overlap.xslice.start:
                xstarts[overlap.xslice.start].append(overlap.fov)
            if overlap.xslice.stop:
                xstops[2048 + overlap.xslice.stop].append(overlap.fov)
            if overlap.yslice.start:
                ystarts[overlap.yslice.start].append(overlap.fov)
            if overlap.yslice.stop:
                ystops[2048 + overlap.yslice.stop].append(overlap.fov)
    for xstart, fovs in xstarts.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs)) & (barcodes["x"] > xstart), "status"
        ] = "edge"
    for ystart, fovs in ystarts.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs)) & (barcodes["y"] > ystart), "status"
        ] = "edge"
    for xstop, fovs in xstops.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs)) & (barcodes["x"] < xstop), "status"
        ] = "edge"
    for ystop, fovs in ystops.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs)) & (barcodes["y"] < ystop), "status"
        ] = "edge"


def create_cell_by_gene_table(barcodes) -> pd.DataFrame:
    # Create cell by gene table
    accepted = barcodes[barcodes["status"] == "good"]
    ctable = pd.crosstab(index=accepted["cell_id"], columns=accepted["gene"])
    # Drop blank barcodes
    drop_cols = [col for col in ctable.columns if "notarget" in col or "blank" in col]
    ctable = ctable.drop(columns=drop_cols)
    return ctable


def assign_to_cells_old(barcodes, masks, drifts=None):
    def clip(x, l, u):
        return l if x < l else u if x > u else x

    def get_cell_id(row):
        fov = int(row["fov"])
        x = int(round(row["x"] + xdrift))
        y = int(round(row["y"] + ydrift))
        try:
            return masks[fov][y, x]
        except IndexError:
            return masks[fov][clip(y, 0, 2047), clip(x, 0, 2047)]

    def get_cell_id_3d(row):
        fov = int(row["fov"])
        # x = int(round(row["x"] + xdrift))
        # y = int(round(row["y"] + ydrift))
        # z = int(round(row["z"]))
        # x = int(round((row["x"] + xdrift) / 4))
        # y = int(round((row["y"] + ydrift) / 4))
        # z = int(round((row["z"]) / 6.3333333))
        x = int(round((row["x"] + xdrift) / 2))
        y = int(round((row["y"] + ydrift) / 2))
        z = int(round((row["z"]) / 2))
        try:
            return masks[fov][z, y, x]
        except IndexError:
            # return masks[fov][z, clip(y, 0, 2047), clip(x, 0, 2047)]
            return masks[fov][
                clip(z, 0, masks[0].shape[0] - 1),
                clip(y, 0, masks[0].shape[1] - 1),
                clip(x, 0, masks[0].shape[2] - 1),
            ]

    masks.link_cells_in_overlaps()
    cellids = []
    for fov, group in tqdm(barcodes.groupby("fov"), desc="Assigning barcodes to cells"):
        if drifts is not None:
            xdrift = drifts.loc[fov]["X drift"]
            ydrift = drifts.loc[fov]["Y drift"]
        else:
            xdrift, ydrift = 0, 0
        if len(masks[0].shape) == 2:
            cellids.append(group.apply(get_cell_id, axis=1))
        elif len(masks[0].shape) == 3:
            cellids.append(group.apply(get_cell_id_3d, axis=1))
    barcodes["cell_id"] = pd.concat(cellids)


class Barcodes:
    def __init__(self, mfx):
        self.mfx = mfx
        self.barcodes = mfx.unassigned_barcodes.copy()
        self.barcodes["status"] = "good"

        # TODO: Automatically determine trim margin parameters
        self.trim_barcodes_on_margins(left=99, right=99, top=99, bottom=99)

        # While this is fairly time consuming, we do it on construction because most of the methods of
        # this class should be used with assigned barcodes
        assign_to_cells(
            self.barcodes, masks=self.mfx.masks
        )  # , drifts=self.mfx.mask_drifts)

        self.filter_barcodes()

        calculate_global_coordinates(self.barcodes, self.mfx.positions)

    def __len__(self):
        return len(self.barcodes)

    @cached_property
    def in_cells(self):
        return len(self.barcodes[self.barcodes["cell_id"] != 0])

    def filter_barcodes(self):
        self.barcodes.loc[
            ~self.barcodes["cell_id"].isin(
                self.mfx.celldata[self.mfx.celldata["status"] == "ok"].index
            ),
            "status",
        ] = "bad cell"
        self.barcodes.loc[self.barcodes["cell_id"] == 0, "status"] = "no cell"

    @cached_property
    def cell_count(self):
        return len(
            np.unique(
                self.barcodes[
                    self.barcodes["cell_id"].isin(
                        self.mfx.celldata[self.mfx.celldata["status"] == "ok"].index
                    )
                ]["cell_id"]
            )
        )

    def trim_barcodes_on_margins(self, left=0, right=0, top=0, bottom=0):
        self.barcodes.loc[self.barcodes["x"] < left, "status"] = "edge"
        self.barcodes.loc[self.barcodes["y"] < top, "status"] = "edge"
        self.barcodes.loc[self.barcodes["x"] > 2048 - right, "status"] = "edge"
        self.barcodes.loc[self.barcodes["y"] > 2048 - bottom, "status"] = "edge"
