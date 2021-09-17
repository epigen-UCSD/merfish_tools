from functools import cached_property
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class Barcodes:
    def __init__(self, mfx):
        self.mfx = mfx
        self.barcodes = mfx.unassigned_barcodes.copy()
        self.barcodes["status"] = "good"

        # TODO: Automatically determine trim margin parameters
        self.trim_barcodes_on_margins(left=99, right=99, top=99, bottom=99)

        # While this is fairly time consuming, we do it on construction because most of the methods of
        # this class should be used with assigned barcodes
        self.assign_barcodes_to_cells(masks=self.mfx.masks, drifts=self.mfx.mask_drifts)

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
            cellids.append(group.apply(get_cell_id, axis=1))
        self.barcodes["cell_id"] = pd.concat(cellids)
