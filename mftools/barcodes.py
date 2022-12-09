"""A collection of functions to work with the barcodes decoded by MERlin.

Functions
---------
process_merlin_barcodes
    Determines error bit/type for barcodes.
expand_codebook
    Creates a new codebook with additional barcodes representing all possible
    single bit flips for all genes. Used to assign error correction information
    to barcodes.
normalize_codebook
    Returns a codebook with L2 normalized barcodes.
set_barcode_stats
    Adds barcode statistics to the global stats object (see stats.py)
make_table
    Given a MERlin analysis folder and codebook, creates a table of all decoded
    barcodes with error correction information.
calculate_global_coordinates
    Adds the global coordinates to a barcode table.
assign_to_cells
    Determines the cell IDs for each barcode.
link_cell_ids
    Renames cell IDs to unify overlapping cells in adjacent FOVs.
mark_barcodes_in_overlaps
    Sets the status of barcodes to "edge" for those barcodes which are in the
    overlapping regions of FOVs.
create_cell_by_gene_table
    Given a barcode table with assigned cell IDs, returns a cell by gene matrix.
    Barcodes marked with "edge" status are not counted.
count_unfiltered_barcodes
    Gets the number of barcodes MERlin decoded before applying adaptive filtering.
get_per_bit_stats
    Calculates the per-bit error rates for a given gene.
get_per_gene_error
    Calculates the overall error rates for each gene.
per_bit_error
    Calculates the average error rate for each bit across all genes.
per_fov_error
    Calculates the error rate within each FOV.
"""
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import norm

from . import fileio
from . import stats
from . import config


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
    try:
        df["error_bit"] = (
            res["id"].str.split("flip", expand=True)[1].fillna(0).astype(int)
        )
    except KeyError:
        df["error_bit"] = 0
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


def set_barcode_stats(
    merlin_result: fileio.MerlinOutput, bcs: pd.DataFrame, colors: list
) -> pd.DataFrame:
    stats.set("Unfiltered barcode count", count_unfiltered_barcodes(merlin_result))
    stats.set("Filtered barcode count", len(bcs))
    stats.set(
        "Unfiltered barcodes per FOV",
        stats.get("Unfiltered barcode count") / stats.get("FOVs"),
    )
    stats.set(
        "Filtered barcodes per FOV",
        stats.get("Filtered barcode count") / stats.get("FOVs"),
    )
    stats.set(
        "% barcodes kept",
        stats.get("Filtered barcode count") / stats.get("Unfiltered barcode count"),
    )
    stats.set("Exact barcode count", len(bcs[bcs["error_type"] == 0]))
    stats.set(
        "Corrected barcode count",
        stats.get("Filtered barcode count") - stats.get("Exact barcode count"),
    )
    stats.set(
        "% exact barcodes",
        stats.get("Exact barcode count") / stats.get("Filtered barcode count"),
    )
    stats.set(
        "0->1 error rate",
        len(bcs[bcs["error_type"] == 1]) / stats.get("Filtered barcode count"),
    )
    stats.set(
        "1->0 error rate",
        len(bcs[bcs["error_type"] == -1]) / stats.get("Filtered barcode count"),
    )
    try:
        error = per_bit_error(bcs, colors)
        stats.set(
            "Average per-bit 0->1 error rate",
            error[error["Error type"] == "0->1"]["Error rate"].mean(),
        )
        stats.set(
            "Average per-bit 1->0 error rate",
            error[error["Error type"] == "1->0"]["Error rate"].mean(),
        )
        return error
    except ValueError:  # Temporary hack to avoid issue
        return None


def make_table(
    merlin_result: fileio.MerlinOutput, codebook: pd.DataFrame
) -> pd.DataFrame:
    """Create a table of all barcodes with error correction information."""
    codebook = expand_codebook(codebook)
    X = np.ascontiguousarray(normalize_codebook(codebook).to_numpy(), dtype=np.float32)
    neighbors = faiss.IndexFlatL2(X.shape[1])
    neighbors.add(X)
    dfs = []
    for fov in tqdm(range(merlin_result.n_fovs()), desc="Preparing barcodes"):
        barcodes = merlin_result.load_filtered_barcodes(fov)
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

    barcodes[["global_x", "global_y"]] = barcodes.groupby("fov", group_keys=False).apply(
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
        x = round(group["x"] + xdrift).apply(lambda n: min(n, 2047)) // config.get(
            "scale"
        )
        y = round(group["y"] + ydrift).apply(lambda n: min(n, 2047)) // config.get(
            "scale"
        )
        x = x.astype(int)
        y = y.astype(int)
        if len(masks[fov].shape) == 3:
            # TODO: Remove hard-coding of scale
            z = round(group["z"] / 6.333333).astype(int)
            cellids.append(pd.Series(masks[fov][z, y, x], index=group.index))
        else:
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
    stats.set("Barcodes assigned to cells", len(barcodes[barcodes["status"] == "good"]))
    stats.set(
        "% barcodes assigned to cells",
        stats.get("Barcodes assigned to cells")
        / len(barcodes[barcodes["status"] != "edge"]),
    )


def link_cell_ids(barcodes, cell_links):
    link_map = {cell: list(group)[0] for group in cell_links for cell in group}
    barcodes["cell_id"] = barcodes["cell_id"].apply(
        lambda cid: link_map[cid] if cid in link_map else cid
    )
    stats.set("Cells with barcodes", len(np.unique(barcodes["cell_id"])))
    stats.set(
        "% cells with barcodes",
        stats.get("Cells with barcodes") / stats.get("Segmented cells"),
    )


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
                xstops[2048 // config.get("scale") + overlap.xslice.stop].append(
                    overlap.fov
                )
            if overlap.yslice.start:
                ystarts[overlap.yslice.start].append(overlap.fov)
            if overlap.yslice.stop:
                ystops[2048 // config.get("scale") + overlap.yslice.stop].append(
                    overlap.fov
                )
    for xstart, fovs in xstarts.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs))
            & (barcodes["x"] // config.get("scale") > xstart),
            "status",
        ] = "edge"
    for ystart, fovs in ystarts.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs))
            & (barcodes["y"] // config.get("scale") > ystart),
            "status",
        ] = "edge"
    for xstop, fovs in xstops.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs))
            & (barcodes["x"] // config.get("scale") < xstop),
            "status",
        ] = "edge"
    for ystop, fovs in ystops.items():
        barcodes.loc[
            (barcodes["fov"].isin(fovs))
            & (barcodes["y"] // config.get("scale") < ystop),
            "status",
        ] = "edge"


def create_cell_by_gene_table(barcodes, drop_blank=True) -> pd.DataFrame:
    # Create cell by gene table
    accepted = barcodes[barcodes["status"] == "good"]
    ctable = pd.crosstab(index=accepted["cell_id"], columns=accepted["gene"])
    # Drop blank barcodes
    if drop_blank:
        drop_cols = [
            col for col in ctable.columns if "notarget" in col or "blank" in col.lower()
        ]
        ctable = ctable.drop(columns=drop_cols)
    stats.set("Median transcripts per cell", np.median(ctable.apply(np.sum, axis=1)))
    stats.set(
        "Median genes detected per cell", np.median(ctable.astype(bool).sum(axis=1))
    )
    return ctable


def count_unfiltered_barcodes(merlin_result: fileio.MerlinOutput) -> int:
    """Count the total number of barcodes decoded by MERlin before adaptive filtering."""
    raw_count = 0
    for fov in tqdm(range(merlin_result.n_fovs()), desc="Counting unfiltered barcodes"):
        raw_count += merlin_result.count_raw_barcodes(fov)
    return raw_count


def get_per_bit_stats(gene: str, barcodes: pd.DataFrame) -> pd.DataFrame:
    k0 = len(barcodes[barcodes["error_type"] == 0])
    total = len(barcodes)
    rows = []
    for bit in range(1, 23):
        errors = barcodes[barcodes["error_bit"] == bit]
        k1 = len(errors)
        if k1 > 0 and k0 > 0:
            err_type = "1->0" if errors.iloc[0]["error_type"] == -1 else "0->1"
            rate = (k1 / k0) / (1 + (k1 / k0))
            rows.append([gene, bit, k1, err_type, rate, rate * total, total])
    return pd.DataFrame(
        rows,
        columns=[
            "gene",
            "Bit",
            "count",
            "Error type",
            "Error rate",
            "weighted",
            "total",
        ],
    ).reset_index()


def per_gene_error(barcodes) -> pd.DataFrame:
    """Get barcode error statistics per gene."""
    error = (
        barcodes.groupby(["gene", "error_type"])
        .count()
        .reset_index()
        .pivot(index="gene", columns="error_type", values="barcode_id")
    )
    error.columns = ["1->0 errors", "Exact barcodes", "0->1 errors"]
    error["Total barcodes"] = (
        error["1->0 errors"] + error["0->1 errors"] + error["Exact barcodes"]
    )
    error["% exact barcodes"] = error["Exact barcodes"] / error["Total barcodes"]
    return error


def per_bit_error(barcodes, colors) -> pd.DataFrame:
    """Get barcode error statistics per bit."""
    error = pd.concat(
        get_per_bit_stats(gene, group) for gene, group in barcodes.groupby("gene")
    )
    error["Color"] = error.apply(
        lambda row: colors[(row["Bit"] - 1) % len(colors)], axis=1
    )
    error["Hybridization round"] = error.apply(
        lambda row: (row["Bit"] - 1) // len(colors) + 1, axis=1
    )
    return error


def per_fov_error(barcodes) -> pd.DataFrame:
    """Get barcode error statistics per FOV."""
    error = (
        barcodes.groupby(["fov", "error_type"]).count()["gene"]
        / barcodes.groupby("fov").count()["gene"]
    )
    error = error.reset_index()
    error.columns = ["FOV", "Error type", "Error rate"]
    error["Error type"] = error["Error type"].replace(
        [0, -1, 1], ["Correct", "1->0", "0->1"]
    )
    return error
