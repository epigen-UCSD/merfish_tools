import os
import glob
from functools import cached_property

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from stats import Stats
import segmentation
from util import expand_codebook, csv_cached_property, calculate_drift
from daxfile import DaxFile
from barcodes import Barcodes
from cellgene import ScanpyObject


class MerfishExperiment:
    """This is the main class for the MERFISH pipeline."""

    def __init__(self):
        self.name = config.get('experiment_name')
        self.analysis_folder = os.path.join(config.get('analysis_root'), self.name)
        self.segmask_folder = os.path.join(config.get('segmentation_root'), self.name, config.get('segmentation_name'))
        self.data_folder = os.path.join(config.get('data_root'), self.name)
        self.mfx = self  # This is dumb, but makes decorators that expect self.mfx to work

    @cached_property
    def stats(self):
        return Stats(self)

    @cached_property
    def raw_barcode_files(self):
        return glob.glob(os.path.join(self.analysis_folder, "Decode", "barcodes", "barcode_data_*.h5"))

    @cached_property
    def filtered_barcode_files(self):
        return glob.glob(os.path.join(self.analysis_folder, "AdaptiveFilterBarcodes", "barcodes", "barcode_data_*.h5"))

    @cached_property
    def barcodes(self):
        return Barcodes(self)

    @cached_property
    def data_organization(self):
        return pd.read_csv(os.path.join(self.analysis_folder, "dataorganization.csv"))

    @cached_property
    def barcode_colors(self):
        # We build the list like this instead of just np.unique on the column to maintain the ordering
        colors = []
        for color in self.data_organization['color']:
            if color not in colors:
                colors.append(color)
        return colors

    @cached_property
    def hyb_drifts(self):
        rows = []
        files = glob.glob(os.path.join(self.analysis_folder, "FiducialCorrelationWarp", "transformations", "offsets_*.npy"))
        for fov, filename in enumerate(files):
            drifts = np.load(filename, allow_pickle=True)
            for hyb, drift in enumerate(drifts[len(self.barcode_colors)::len(self.barcode_colors)], start=2):
                rows.append([fov, hyb, drift.params[0][2], drift.params[1][2]])
        return pd.DataFrame(rows, columns=["FOV", "Hybridization round", "X drift", "Y drift"])

    @csv_cached_property('mask_drifts.csv')
    def mask_drifts(self):
        # TODO: Filename is hardcoded to 3 digit FOV numbers
        # TODO: Number of channels in H0 and H1 images are hardcoded
        # TODO: Fiducial channel to use is hardcoded
        drifts = []
        for fov in tqdm(range(len(self.masks)), desc="Calculating drifts between barcodes and masks"):
            h0 = DaxFile(os.path.join(self.data_folder, f'Conv_zscan_H0_F_{fov:03d}.dax'), num_channels=5).zslice(0, channel=2)
            h1 = DaxFile(os.path.join(self.data_folder, f'Conv_zscan_H1_F_{fov:03d}.dax'), num_channels=3).zslice(0, channel=2)
            drifts.append(calculate_drift(h0, h1))
        driftdf = pd.concat(drifts, axis=1).T
        driftdf.columns = ['Y drift', 'X drift']
        driftdf = driftdf.fillna(0)
        return driftdf

    @cached_property
    def codebook(self):
        return pd.read_csv(glob.glob(os.path.join(self.analysis_folder, "codebook_*.csv"))[0])

    @csv_cached_property('expanded_codebook.csv')
    def expanded_codebook(self):
        return expand_codebook(self.codebook)

    @cached_property
    def positions(self):
        df = pd.read_csv(os.path.join(self.analysis_folder, "positions.csv"), header=None)
        df.columns = ['x', 'y']
        return df

    @cached_property
    def masks(self):
        return segmentation.MaskList(mfx=self, segmask_dir=self.segmask_folder)

    @csv_cached_property('cell_metadata.csv')
    def celldata(self):
        celldata = self.masks.create_metadata_table(use_overlaps=True)
        celldata = segmentation.filter_by_volume(celldata, min_volume=config.get('minimum_cell_volume'),
                                                 max_factor=config.get('maximum_cell_volume'))
        return celldata

    @csv_cached_property('global_cell_positions.csv', save_index=True, index_col=0)
    def global_cell_positions(self):
        return segmentation.get_global_cell_positions(self.celldata, self.positions)

    @csv_cached_property('single_cell_raw_counts.csv', save_index=True, index_col=0)
    def single_cell_raw_counts(self):
        return self.barcodes.cell_by_gene_table

    @cached_property
    def clustering(self):
        return ScanpyObject(self)
