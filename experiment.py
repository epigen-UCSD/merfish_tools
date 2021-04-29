import os
import glob
from functools import cached_property

import numpy as np
import pandas as pd

import config
from stats import Stats
from util import announce, expand_codebook, csv_cached_property


class MerfishExperiment:
    """
    This is the main class for the MERFISH pipeline.
    """
    def __init__(self):
        #self.args = args
        self.name = config.get('experiment_name')
        self.analysis_folder = os.path.join(config.get('analysis_root'), self.name)
        self.mfx = self #This is dumb, but makes decorators that expect self.mfx to work

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
    @announce("Counting filtered barcodes")
    def barcodes(self):
        return pd.read_csv(os.path.join(self.analysis_folder, "ExportBarcodes", "barcodes.csv"))

    @cached_property
    def data_organization(self):
        return pd.read_csv(os.path.join(self.analysis_folder, "dataorganization.csv"))

    @cached_property
    def barcode_colors(self):
        #We build the list like this instead of just np.unique on the column to maintain the ordering
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
