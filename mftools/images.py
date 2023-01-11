"""Functions for working with fluorescent microscopy images."""

import math
import random
import functools
from typing import List
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from . import fileio


Overlap = namedtuple("Overlap", ["fov", "xslice", "yslice"])


def get_slice(diff: float, fovsize: int = 220, get_trim: bool = False) -> slice:
    """Get a slice for the region of an image overlapped by another FOV.

    :param diff: The amount of overlap in the global coordinate system.
    :param fovsize: The width/length of a FOV in the global coordinate system, defaults to 220.
    :param get_trim: If True, return the half of the overlap closest to the edge. This is for
        determining in which region the barcodes should be trimmed to avoid duplicates.
    :return: A slice in the FOV coordinate system for the overlap.
    """
    if int(diff) == 0:
        return slice(None)
    if diff > 0:
        if get_trim:
            diff = fovsize - ((fovsize - diff) / 2)
        overlap = 2048 * diff / fovsize
        return slice(math.trunc(overlap), None)
    else:
        if get_trim:
            diff = -fovsize - ((-fovsize - diff) / 2)
        overlap = 2048 * diff / fovsize
        return slice(None, math.trunc(overlap))


def lowpass_filter(image, sigma, window_size=None):
    if not window_size:
        window_size = int(2 * np.ceil(2 * sigma) + 1)
    if image.ndim == 3:
        return np.array([lowpass_filter(zslice, sigma, window_size) for zslice in image])
    return cv2.GaussianBlur(image, (window_size, window_size), sigma, borderType=cv2.BORDER_REPLICATE)


def highpass_filter(image, sigma, window_size=None):
    if image.ndim == 3:
        return np.array([highpass_filter(zslice, sigma, window_size) for zslice in image])
    blur = lowpass_filter(image, sigma, window_size)
    filtered = image - blur
    filtered[blur > image] = 0
    return filtered


class FOVPositions:
    def __init__(
        self, positions: pd.DataFrame = None, filename: str = None, merlin: fileio.MerlinOutput = None
    ) -> None:
        if positions is not None:
            self.positions = positions
        elif filename is not None:
            self.positions = fileio.load_fov_positions(filename)
        elif merlin is not None:
            self.positions = merlin.load_fov_positions()

    @functools.cached_property
    def overlaps(self):
        return self.find_fov_overlaps()

    def local_to_global_coordinates(self, x, y, fov):
        global_x = 220 * x / 2048 + np.array(self.positions.loc[fov]["y"])
        global_y = 220 * y / 2048 - np.array(self.positions.loc[fov]["x"])
        return global_x, global_y

    def find_fov_overlaps(self, fovsize: int = 220, get_trim: bool = False) -> List[list]:
        """Identify overlaps between FOVs."""
        neighbor_graph = NearestNeighbors()
        neighbor_graph = neighbor_graph.fit(self.positions)
        res = neighbor_graph.radius_neighbors(self.positions, radius=fovsize, return_distance=True, sort_results=True)
        overlaps = []
        pairs = set()
        for i, (dists, fovs) in enumerate(zip(*res)):
            i = self.positions.iloc[i].name
            for dist, fov in zip(dists, fovs):
                fov = self.positions.iloc[fov].name
                if dist == 0 or (i, fov) in pairs:
                    continue
                pairs.update([(i, fov), (fov, i)])
                diff = self.positions.loc[i] - self.positions.loc[fov]
                _get_slice = functools.partial(get_slice, fovsize=fovsize, get_trim=get_trim)
                overlaps.append(
                    [
                        Overlap(i, _get_slice(diff[0]), _get_slice(-diff[1])),
                        Overlap(fov, _get_slice(-diff[0]), _get_slice(diff[1])),
                    ]
                )
        return overlaps


def get_median_image(imageset: fileio.ImageDataset, bit: int, sample_size: int = None) -> np.ndarray:
    fovlist = list(range(imageset.n_fovs()))
    random.shuffle(fovlist)
    medimg = np.array([imageset.load_image(fov, zslice=10, bit=bit) for fov in tqdm(fovlist[:sample_size])])
    return np.median(medimg, axis=0)


def flat_field_correct(image: np.ndarray, sigma: float, filter_size: int = None) -> np.ndarray:
    if filter_size is None:
        filter_size = int(2 * np.ceil(2 * sigma) + 1)
    blur = cv2.blur(image, (filter_size, filter_size), sigma)
    return image / blur
