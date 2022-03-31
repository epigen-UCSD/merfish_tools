from functools import cached_property

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import norm

import fileio


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


def make_table(analysis_dir: str, codebook: pd.DataFrame) -> pd.DataFrame:
    """Create a table of all barcodes with error correction information."""
    codebook = expand_codebook(codebook)
    X = np.ascontiguousarray(normalize_codebook(codebook).to_numpy(), dtype=np.float32)
    neighbors = faiss.IndexFlatL2(X.shape[1])
    neighbors.add(X)
    dfs = []
    for barcodes in tqdm(
        fileio.merlin_barcodes(analysis_dir), desc="Preparing barcodes"
    ):
        dfs.append(process_merlin_barcodes(barcodes, neighbors, codebook))
    df = pd.concat(dfs, ignore_index=True)
    df["status"] = "Unprocessed"
    return df


class Barcodes:
    def __init__(self, mfx):
        self.mfx = mfx
        self.barcodes = mfx.unassigned_barcodes.copy()
        self.barcodes["status"] = "good"

        # TODO: Automatically determine trim margin parameters
        self.trim_barcodes_on_margins(left=99, right=99, top=99, bottom=99)

        # While this is fairly time consuming, we do it on construction because most of the methods of
        # this class should be used with assigned barcodes
        self.assign_barcodes_to_cells(
            masks=self.mfx.masks
        )  # , drifts=self.mfx.mask_drifts)

        self.filter_barcodes()

        self.calculate_global_coordinates(self.mfx.positions)

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

    def calculate_global_coordinates(self, positions):
        def convert_to_global(group):
            fov = int(group.iloc[0]["fov"])
            ypos = positions.loc[fov][0]
            xpos = positions.loc[fov][1]
            group["global_y"] = 220 * (2048 - group["y"]) / 2048 + ypos
            group["global_x"] = 220 * group["x"] / 2048 + xpos
            return group[["global_x", "global_y"]]

        self.barcodes[["global_x", "global_y"]] = self.barcodes.groupby("fov").apply(
            convert_to_global
        )

    def assign_barcodes_to_cells(self, masks, drifts=None):
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
            x = int(round((row["x"] + xdrift) / 4))
            y = int(round((row["y"] + ydrift) / 4))
            z = int(round((row["z"]) / 6.3333333))
            try:
                return masks[fov][z, y, x]
            except IndexError:
                # return masks[fov][z, clip(y, 0, 2047), clip(x, 0, 2047)]
                return masks[fov][z, clip(y, 0, 511), clip(x, 0, 511)]

        masks.link_cells_in_overlaps()
        cellids = []
        for fov, group in tqdm(
            self.barcodes.groupby("fov"), desc="Assigning barcodes to cells"
        ):
            if drifts is not None:
                xdrift = drifts.loc[fov]["X drift"]
                ydrift = drifts.loc[fov]["Y drift"]
            else:
                xdrift, ydrift = 0, 0
            if len(masks[0].shape) == 2:
                cellids.append(group.apply(get_cell_id, axis=1))
            elif len(masks[0].shape) == 3:
                cellids.append(group.apply(get_cell_id_3d, axis=1))
        self.barcodes["cell_id"] = pd.concat(cellids)
