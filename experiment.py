"""This module is the entry point for using this Merfish analysis package.

This Merfish analysis software can be used on the command-line using the run.py script,
however this module and the MerfishExperiment class are the entry point for using
this package in other python scripts by creating and using a MerfishExperiment object.
"""
import os
import glob
import typing
from functools import cached_property
from matplotlib.pyplot import bar

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import norm
from sklearn.neighbors import NearestNeighbors

import config
from stats import Stats
import segmentation
from util import expand_codebook, csv_cached_property, calculate_drift
from daxfile import DaxFile
from barcodes import Barcodes
from cellgene import ScanpyObject


class MerfishExperiment:
    """This is the main class for the MERFISH pipeline."""

    def __init__(self) -> None:
        """Initialize the Experiment object using paths from the config."""
        self.name = config.get("experiment_name")
        self.analysis_folder = os.path.join(config.get("analysis_root"), self.name)
        self.segmask_folder = os.path.join(
            config.get("segmentation_root"), self.name, config.get("segmentation_name")
        )
        self.data_folder = os.path.join(config.get("data_root"), self.name)
        self.mfx = (
            self  # This is dumb, but makes decorators that expect self.mfx to work
        )

    @cached_property
    def stats(self) -> Stats:
        """Get a Stats object (see stats.py) that calculates various metrics."""
        return Stats(self)

    @cached_property
    def fovs(self) -> typing.List[int]:
        df = pd.read_csv(
            os.path.join(self.analysis_folder, "positions.csv"), header=None
        )
        return [i for i in range(len(df)) if i not in config.get("omit_fovs")]

    @cached_property
    def raw_barcode_files(self) -> typing.List[str]:
        """Get the list of raw barcode files produced by MERlin."""
        folder = os.path.join(self.analysis_folder, "Decode", "barcodes")
        return [os.path.join(folder, f"barcode_data_{fov}.h5") for fov in self.fovs]

    @cached_property
    def filtered_barcode_files(self) -> typing.List[str]:
        """Get the list of filtered barcode files produced by MERlin."""
        folder = os.path.join(
            self.analysis_folder, "AdaptiveFilterBarcodes", "barcodes"
        )
        return [os.path.join(folder, f"barcode_data_{fov}.h5") for fov in self.fovs]

    @cached_property
    def data_organization(self) -> pd.DataFrame:
        """Get the data organization table used by MERlin.

        The data organization table has a row for each MERFISH encoding bit, and the columns
        contain information about the filenames for the raw images and how the various bits
        are stored, e.g. which channels each bit is in and what color those channels are.
        See the MERlin documentation for more information.
        """
        return pd.read_csv(os.path.join(self.analysis_folder, "dataorganization.csv"))

    @cached_property
    def barcode_colors(self) -> typing.List[str]:
        """Get the list of colors used for encoding bits in this experiment.

        This does not include colors used for fiducial beads, DAPI staining, etc.
        """
        # We don't just use np.unique on the column so that we can maintain the ordering
        colors = []
        for color in self.data_organization["color"]:
            if color not in colors:
                colors.append(color)
        return colors

    @cached_property
    def hyb_drifts(self) -> pd.DataFrame:
        """Get the drifts calculated between hybridization rounds.

        The 'X drift' and 'Y drift' columns indicate the translation required to
        align coordinates in the FOV and hybridization round to the first hybridization
        round for that FOV. These drifts are calculated by MERlin.
        """
        rows = []
        folder = os.path.join(
            self.analysis_folder, "FiducialBeadWarp", "transformations"
        )
        files = [os.path.join(folder, f"offsets_{fov}.npy") for fov in self.fovs]
        num_colors = len(self.barcode_colors)
        for fov, filename in enumerate(files):
            drifts = np.load(filename, allow_pickle=True)
            for hyb, drift in enumerate(drifts[num_colors::num_colors], start=2):
                rows.append([fov, hyb, drift.params[0][2], drift.params[1][2]])
        return pd.DataFrame(
            rows, columns=["FOV", "Hybridization round", "X drift", "Y drift"]
        )

    @csv_cached_property("mask_drifts.csv", save_index=True)
    def mask_drifts(self) -> pd.DataFrame:
        """Get the drifts between the DAPI/polyA staining and first hybridization round.

        The 'X drift' and 'Y drift' columns indicate the translation required to align
        coordinates in the first hybridization round to the image taken prior to round
        1, used to generate the segmentation mask. The index of the DataFrame is the FOV.
        """
        # TODO: Filename is hardcoded to 3 digit FOV numbers
        # TODO: Number of channels in H0 and H1 images are hardcoded
        # TODO: Fiducial channel to use is hardcoded
        drifts = []
        for fov in tqdm(
            self.fovs, desc="Calculating drifts between barcodes and masks"
        ):
            h0 = DaxFile(
                os.path.join(self.data_folder, f"Conv_zscan_H0_F_{fov:03d}.dax"),
                num_channels=5,
            ).zslice(0, channel=2)
            h1 = DaxFile(
                os.path.join(self.data_folder, f"Conv_zscan_H1_F_{fov:03d}.dax"),
                num_channels=3,
            ).zslice(0, channel=2)
            drifts.append(calculate_drift(h0, h1))
        driftdf = pd.concat(drifts, axis=1).T
        driftdf.columns = ["Y drift", "X drift"]
        driftdf = driftdf.fillna(0)
        driftdf["FOV"] = self.fovs
        driftdf = driftdf.set_index("FOV")
        return driftdf

    @cached_property
    def codebook(self) -> pd.DataFrame:
        """Get the codebook used for this MERFISH experiment.

        The 'name' and 'id' columns are identical, and both contain the name of the
        gene or blank barcode encoded by that row. The 'bit1' through 'bitN' columns
        contain the 0s or 1s of the barcode.
        """
        return pd.read_csv(
            glob.glob(os.path.join(self.analysis_folder, "codebook_*.csv"))[0]
        )

    @csv_cached_property("expanded_codebook.csv")
    def expanded_codebook(self) -> pd.DataFrame:
        """Get the codebook expanded with single bit errors.

        This codebook contains all the rows in the normal codebook, plus an additional 16
        rows per barcode which represent all possible single-bit errors. The 'name' column
        contains the name of the original barcode, while the 'id' column is a unique
        name indicating which bit was flipped, e.g. 'MALAT1_flip3'. Therefore, there
        should be 17 rows with the same 'name': the original barcode and all 16 possible
        single-bit errors, while every row should have a unique 'id'.
        """
        return expand_codebook(self.codebook)

    @cached_property
    def positions(self) -> pd.DataFrame:
        """Get the global positions of the FOVs.

        The coordinates indicate the top-left corner of the FOV.
        """
        df = pd.read_csv(
            os.path.join(self.analysis_folder, "positions.csv"), header=None
        )
        df.columns = ["x", "y"]
        df = df.reindex(index=self.fovs)  # Drop omitted FOVs
        return df

    @cached_property
    def masks(self) -> segmentation.MaskList:
        """Get the segmentation masks."""
        return segmentation.MaskList(mfx=self, segmask_dir=self.segmask_folder)

    @csv_cached_property("unassigned_barcodes.csv")
    def unassigned_barcodes(self) -> pd.DataFrame:
        """Get the table of filtered barcodes output by MERlin.

        This does not load the table in the FilteredBarcodes folder from MERlin, but rather
        constructs a new table from the barcode data in AdaptiveFilteredBarcodes. This is
        so the barcodes can be annotated with error correction information.
        """
        codebook = self.mfx.expanded_codebook
        codes = codebook.filter(like="bit")
        normcodes = codes.apply(lambda row: row / norm(row), axis=1)
        X = np.ascontiguousarray(normcodes.to_numpy(), dtype=np.float32)
        neighbors = faiss.IndexFlatL2(X.shape[1])
        neighbors.add(X)
        dfs = []
        for fov, barcode_file in tqdm(
            list(zip(self.mfx.fovs, self.mfx.filtered_barcode_files)),
            desc="Preparing barcodes",
        ):
            barcodes = pd.read_hdf(barcode_file)

            # Find nearest
            X = np.ascontiguousarray(
                barcodes.filter(like="intensity_").to_numpy(), dtype=np.float32
            )
            indexes = neighbors.search(X, k=1)

            res = codebook.iloc[indexes[1].flatten()].copy()
            res["fov"] = fov
            res = res.set_index(["name", "id", "fov"]).filter(like="bit").sum(axis=1)
            res = pd.DataFrame(res, columns=["bits"]).reset_index()

            testdf = barcodes[["barcode_id", "fov", "x", "y", "z"]]
            testdf = testdf.reset_index(drop=True)
            testdf["gene"] = res["name"]
            testdf["error_type"] = res["bits"] - 4
            testdf["error_bit"] = res["id"].str.split("flip", expand=True)[1].fillna(0)
            if config.get("flip_barcodes"):
                testdf["x"] = 2048 - testdf["x"]
                testdf["y"] = 2048 - testdf["y"]
            if config.get("transpose_barcodes"):
                testdf["x"], testdf["y"] = testdf["y"], testdf["x"]
            dfs.append(testdf)
        return pd.concat(dfs)

    @csv_cached_property("assigned_barcodes.csv")
    def assigned_barcodes(self) -> pd.DataFrame:
        """Get the table of barcodes assigned to cells, but not filtered."""
        return Barcodes(self).barcodes

    @csv_cached_property("fov_cell_metadata.csv")
    def fov_celldata(self) -> pd.DataFrame:
        """Get the per-fov cell metadata."""
        return self.masks.create_metadata_table(use_overlaps=True)

    @csv_cached_property("cell_metadata.csv", save_index=True)
    def celldata(self) -> pd.DataFrame:
        """Get the global cell metadata.

        The table contains metadata about cells such as their position, volume,
        and filtered status.
        """
        celldata = (
            self.fov_celldata.drop(columns=["fov_y", "fov_x", "fov_volume"])
            .groupby("cell_id")
            .agg({"fov": list, "volume": max})
        )
        global_cell_positions = segmentation.get_global_cell_positions(
            self.fov_celldata, self.positions, masksize=config.get("mask_size")
        )
        celldata = celldata.merge(
            right=global_cell_positions, left_index=True, right_index=True
        )
        celldata["status"] = "ok"
        celldata = segmentation.filter_by_volume(
            celldata,
            min_volume=config.get("minimum_cell_volume"),
            max_factor=config.get("maximum_cell_volume"),
        )
        return celldata

    def save_celldata(self) -> None:
        """Resave the cell metadata table after it has been changed."""
        filename = config.path("cell_metadata.csv")
        self.celldata.to_csv(filename, index=True)

    def update_filtered_celldata(self, status: str) -> None:
        """Update the status of recently filtered cells.

        Look for cells in the cell metadata table with the status 'ok' that
        are no longer present in the scanpy object (due to some recently applied
        filtering step) and change their status to the one given, then save the
        table. This function should be called after doing any type of filtering
        on the scanpy object.
        """
        missing = self.celldata.merge(
            self.single_cell_raw_counts,
            right_index=True,
            left_index=True,
            how="left",
            indicator=True,
        )
        missing = missing[
            (missing["_merge"] == "left_only") & (missing["status"] == "ok")
        ].index
        self.celldata.loc[missing, "status"] = status
        self.save_celldata()

    @csv_cached_property("single_cell_raw_counts.csv", save_index=True)
    def single_cell_raw_counts(self) -> pd.DataFrame:
        """Get the cell-by-gene table of raw molecule counts per cell."""
        # Create cell by gene table
        filtered_barcodes = self.assigned_barcodes[
            self.assigned_barcodes["status"] == "good"
        ]
        ctable = pd.crosstab(
            index=filtered_barcodes.cell_id, columns=filtered_barcodes.gene
        )
        # Drop blank barcodes
        drop_cols = [
            col for col in ctable.columns if "notarget" in col or "blank" in col
        ]
        ctable = ctable.drop(columns=drop_cols)
        # Update cell metadata table. This is identical to update_filtered_celldata,
        # but we can't call it here because we're currently constructing the cell by gene
        # table, so it would cause an infinite recursion.
        missing = self.celldata.merge(
            ctable, right_index=True, left_index=True, how="left", indicator=True
        )
        missing = missing[
            (missing["_merge"] == "left_only") & (missing["status"] == "ok")
        ].index
        self.celldata.loc[missing, "status"] = "No barcodes"
        self.save_celldata()
        return ctable

    @cached_property
    def clustering(self) -> ScanpyObject:
        """Get a ScanpyObject class, which wraps the AnnData object used by scanpy."""
        return ScanpyObject(self)
