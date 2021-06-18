from functools import cached_property
import glob
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class Barcodes:
    def __init__(self, mfx):
        self.mfx = mfx
        print("Loading barcodes...", end='', flush=True)
        barcode_file = os.path.join(self.mfx.analysis_folder, "ExportBarcodes", "barcodes.csv")
        self.barcodes = pd.read_csv(barcode_file)
        print(f"{len(self.barcodes):,d} barcodes found")
        #TODO: Trim barcodes on margins
        #TODO: Remove FOVs in blacklist
        #While this is fairly time consuming, we do it on construction because most of the methods of
        #this class should be used with assigned barcodes
        self.assign_barcodes_to_cells(masks=self.mfx.masks, drifts=self.mfx.mask_drifts)

    def __len__(self):
        return len(self.barcodes)

    @cached_property
    def in_cells(self):
        return len(self.barcodes[self.barcodes['cell_id'] != 0])

    @cached_property
    def filtered_barcodes(self):
        #Drop barcodes not assigned to cells
        filtered_barcodes = self.barcodes[self.barcodes['cell_id'] != 0]
        #Drop barcodes in cells that were pruned
        filtered_barcodes = filtered_barcodes[filtered_barcodes['cell_id'].isin(self.mfx.celldata['cell_id'])]
        return filtered_barcodes

    @cached_property
    def cell_count(self):
        return len(np.unique(self.barcodes[self.barcodes['cell_id'].isin(self.mfx.celldata['cell_id'])]['cell_id']))

    def trim_barcodes_on_margins(self, left=0, right=0, top=0, bottom=0):
        #TODO: Automatically decide on trimming based on drifts
        self.barcodes = self.barcodes[self.barcodes['x'] >= left]
        self.barcodes = self.barcodes[self.barcodes['y'] >= top]
        self.barcodes = self.barcodes[self.barcodes['x'] <= 2048 - right]
        self.barcodes = self.barcodes[self.barcodes['y'] <= 2048 - bottom]

    def drop_fovs(self, fovs):
        self.barcodes = self.barcodes[~self.barcodes['fov'].isin(fovs)]

    def calculate_global_coordinates(self, positions):
        def convert_to_global(group):
            fov = int(group.iloc[0]['fov'])
            ypos = positions.iloc[fov][0]
            xpos = positions.iloc[fov][1]
            group['global_y'] = 220 * (2048 - group['y']) / 2048 + ypos
            group['global_x'] = 220 * group['x'] / 2048 + xpos
            return group[['global_x', 'global_y']]
        self.barcodes[['global_x', 'global_y']] = self.barcodes.groupby('fov').apply(convert_to_global)

    def assign_barcodes_to_cells(self, masks, drifts=None, scale_factor=1):
        clip = lambda x, l, u: l if x < l else u if x > u else x
        def get_cell_id(row):
            fov = int(row['fov'])
            x = int(round(row['x'] + xdrift))
            y = int(round(row['y'] + ydrift))
            try:
                return masks[fov][y, x]
            except IndexError:
                return masks[fov][clip(y, 0, 2047), clip(x, 0, 2047)]

        masks.link_cells_in_overlaps()
        cellids = []
        for fov, group in tqdm(self.barcodes.groupby('fov'), desc="Assigning barcodes to cells"):
            if drifts is not None:
                xdrift = drifts.iloc[fov]['X drift']
                ydrift = drifts.iloc[fov]['Y drift']
            else:
                xdrift, ydrift = 0, 0
            cellids.append(group.apply(get_cell_id, axis=1))
        self.barcodes['cell_id'] = pd.concat(cellids)

    @cached_property
    def cell_by_gene_table(self):
        #Create cell by gene table
        ctable = pd.crosstab(index=self.filtered_barcodes.cell_id, columns=self.filtered_barcodes.barcode_id)
        #Rename barcode ids to gene names
        ctable = ctable.rename(columns=lambda n: self.mfx.codebook.loc[n]['name'])
        #Drop notarget barcodes
        drop_cols = [col for col in ctable.columns if 'notarget' in col]
        ctable = ctable.drop(columns=drop_cols)
        return ctable

    def plot_barcodes_in_fov(self):
        pass
