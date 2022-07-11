"""Functions for processing MERFISH images."""

import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.registration import phase_cross_correlation
from scipy.stats import zscore
import pandas as pd

import fileio


def normalize_brightness(imgs, quants=[25, 35, 45, 55, 65, 75]):
    percentiles = np.array([np.percentile(img, quants) for img in imgs])
    scale_factors = (percentiles.mean(axis=0) / percentiles).mean(axis=1)
    return np.array(
        [img * scale_factor for img, scale_factor in zip(imgs, scale_factors)]
    )


def highpass_filter(img, ksize=(13, 13), sigma=3):
    blur = cv2.GaussianBlur(img.astype(np.float), ksize, sigma, cv2.BORDER_REPLICATE)
    newimg = img - blur
    newimg[blur > img] = 0
    return newimg


def flat_field_correct(img, ksize=(201, 201), sigma=75):
    dst = cv2.GaussianBlur(img, ksize, sigma)
    avg_hist = img.mean()
    return (img / dst) * avg_hist


def calculate_drift(img1, img2, chunks=2):
    inds = np.linspace(0, img1.shape[0], num=chunks + 1, dtype=int)
    drifts = []
    for xstart, xstop in zip(inds, inds[1:]):
        for ystart, ystop in zip(inds, inds[1:]):
            drifts.append(
                phase_cross_correlation(
                    img1[xstart:xstop, ystart:ystop], img2[xstart:xstop, ystart:ystop]
                )[0]
            )
    df = pd.DataFrame(drifts)
    # print(df)
    # print(df.std())
    # Identify quadrants with outliers and remove
    if (df.std() > 15).any():
        df = df[(np.abs(zscore(df)) < 1.25).all(axis=1)]
    # If some quadrants are 0,0 they are probably bad, unless most are 0,0
    # then the non-zero ones are probably bad
    if df.any(axis=1).sum() > len(df) // 4:
        df = df[df.any(axis=1)]
    # print(df)
    return np.array(df.median(), dtype=np.int16)


def align_image(img, drift):
    rolled = np.roll(img, drift, axis=(0, 1))
    # Maybe set edges to 0?
    # Wrapping might not be bad, it will be noise and we throw away
    # molecules on edges anyway
    # i2[-500:, :] = 0
    # i2[:, :80] = 0
    return rolled


def align_stack(imgs, drifts):
    aln = np.array(
        [
            align_image(img, drift)
            for img, drift in zip(imgs[2:], np.repeat(drifts.astype(int), 2, axis=0))
        ]
    )
    return np.concatenate([imgs[:2, :, :], aln])


def load_fiducial_stack(data_dir, fov, rounds=11, channel=2, zslice=0):
    imgs = []
    for round in range(1, rounds + 1):
        path = os.path.join(data_dir, f"Conv_zscan_H{round}_F_{fov:03d}.dax")
        imgs.append(fileio.DaxFile(path, num_channels=3).zslice(zslice, channel))
    return np.array(imgs)


def load_combinatorial_stack(data_dir, fov, zslice, rounds=11, bits_per_round=2):
    imgs = []
    for round in range(1, rounds + 1):
        path = os.path.join(data_dir, f"Conv_zscan_H{round}_F_{fov:03d}.dax")
        dax = fileio.DaxFile(path, num_channels=3)
        for bit in range(0, bits_per_round):
            imgs.append(dax.zslice(zslice, bit))
    return np.array(imgs)


def find_adjacent(mask, id, x, y):
    if x < 0 or y < 0 or x >= mask.shape[0] or y >= mask.shape[1] or mask[x, y] != id:
        return []
    mask[x, y] = -1
    return (
        [(x, y)]
        + find_adjacent(mask, id, x + 1, y)
        + find_adjacent(mask, id, x - 1, y)
        + find_adjacent(mask, id, x, y + 1)
        + find_adjacent(mask, id, x, y - 1)
    )


def L2_normalize_stack(imgs):
    l2 = np.linalg.norm(imgs, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(imgs / l2)


def assign_pixels_to_molecules(decoded):
    molecules = []
    for x in range(decoded.shape[0]):
        for y in range(decoded.shape[1]):
            if decoded[x, y] >= 0:
                molecules.append(
                    (int(decoded[x, y]), find_adjacent(decoded, decoded[x, y], x, y))
                )
    return molecules


def decode_pixels(imgs, codebook_idx, X_codebook):

    l2vecs = L2_normalize_stack(imgs)

    X_pixels = np.moveaxis(np.reshape(l2vecs, (22, 2048 * 2048)), 0, -1)

    # indexes = nn.kneighbors(X_pixels)
    dists, indexes = codebook_idx.search(
        np.ascontiguousarray(X_pixels, dtype=np.float32), k=1
    )
    dists = np.sqrt(dists)

    decoded = np.zeros(X_pixels.shape[0]) - 1
    valid = dists.flatten() <= 0.5167
    decoded[valid] = indexes.flatten()[valid]
    decoded = np.reshape(decoded, imgs[0].shape)

    molecules = assign_pixels_to_molecules(decoded)

    rows = np.zeros((valid.sum(), 9))
    cursor = 0
    for i, molecule in enumerate(molecules):
        end = cursor + len(molecule[1])
        rows[cursor:end, :2] = molecule[1]
        rows[cursor:end, 2] = i
        rows[cursor:end, 3] = molecule[0]
        rows[cursor:end, 4] = len(molecule[1])
        dim1 = np.ravel_multi_index(np.transpose(np.array(molecule[1])), (2048, 2048))
        dim2 = X_codebook[molecule[0]] > 0
        flatimg = np.moveaxis(np.reshape(imgs, (22, 2048 * 2048)), 0, -1)[dim1]
        rows[cursor:end, 5] = flatimg[:, dim2].mean(axis=1)
        rows[cursor:end, 6] = flatimg[:, dim2].max(axis=1)
        rows[cursor:end, 7] = dists[dim1].flatten()
        rows[cursor:end, 8] = flatimg[:, dim2].mean(axis=1) / flatimg.mean(axis=1)
        # rows[cursor:end, 9] = flatimg.max(axis=1)
        # rows[cursor:end, 10] = flatimg.std(axis=1)
        cursor += len(molecule[1])

    pixels = pd.DataFrame(
        rows,
        columns=[
            "x",
            "y",
            "rna_id",
            "barcode_id",
            "area",
            "mean_intensity",
            "max_intensity",
            "distance",
            "snr",
            # "max_intensity_all",
            # "std_intensity",
        ],
    )
    return pixels


def combine_pixels_to_molecules(pixels):
    return pixels.groupby("rna_id").agg(
        {
            "x": "mean",
            "y": "mean",
            "barcode_id": "min",
            "area": "min",
            "mean_intensity": "mean",
            "max_intensity": "max",
            "distance": "min",
            "snr": "mean",
        }
    )


def calc_misid(data):
    return (len(data[data["barcode_id"] < 10]) / 10) / (
        len(data[data["barcode_id"] >= 10]) / 238
    )


def filter_molecules_histogram(molecules):
    histogram = {
        "mean_intensity": np.percentile(molecules["mean_intensity"], range(2, 98, 2)),
        "distance": np.percentile(molecules["distance"], range(2, 98, 2)),
        "area": [1, 2, 3, 4, 5],
    }
    bins = [
        np.searchsorted(edges, molecules[feature])
        for feature, edges in histogram.items()
    ]
    molecules["bin"] = [tuple(x) for x in np.array(bins).T]
    blnkcount = np.unique(
        molecules[molecules["barcode_id"] < 10]["bin"], return_counts=True
    )
    genecount = np.unique(
        molecules[molecules["barcode_id"] >= 10]["bin"], return_counts=True
    )

    bincounts = pd.merge(
        pd.DataFrame(
            blnkcount[1], index=[tuple(row) for row in blnkcount[0]], columns=["c"]
        ),
        pd.DataFrame(
            genecount[1], index=[tuple(row) for row in genecount[0]], columns=["c"]
        ),
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=["blank", "gene"],
    )
    bincounts = bincounts.fillna(0) + 1
    bincounts["ratio"] = bincounts["cblank"] / (
        bincounts["cblank"] + bincounts["cgene"]
    )

    cutoff = len(bincounts) // 2
    top = len(bincounts)
    bottom = 0
    while top - bottom > 1:
        misid = calc_misid(
            molecules[
                molecules["bin"].isin(bincounts.sort_values("ratio").head(cutoff).index)
            ]
        )
        print(misid, cutoff, top, bottom)
        if misid > 0.05:
            top = cutoff
        else:
            bottom = cutoff
        cutoff = (top + bottom) // 2
    return molecules[
        molecules["bin"].isin(bincounts.sort_values("ratio").head(cutoff).index)
    ]
