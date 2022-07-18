"""Functions for processing MERFISH images.

Many functions can take an image "stack" which is expected to be a 4-dimensional
numpy array with the axes corresponding to bit, z, x, and y. For example,
a 22-bit MERFISH experiment imaged with 20 z-slices and 2048x2048 pixel images
would expect the stack to have a shape of (22, 20, 2048, 2048). The function
load_combinatorial_stack loads DAX images from disk and returns such an array.
"""

import os
import logging
from typing import Iterable

import faiss
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import gaussian_filter, label

# from cupyx.scipy.signal import gaussian
from scipy import signal
from cupyx.scipy.fft import fftn, ifftn
from scipy.stats import zscore

import fileio


def gaussian_kernel(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def normalize_brightness(
    stack: cp.ndarray,
    quants: Iterable[int] = (25, 35, 45, 55, 65, 75),
) -> None:
    """Normalize the brightness of a FOV across hybridization rounds.

    Brightness is normalized by calculating the mean brightness at different
    percentiles across the images and then computing a scaling factor for each
    images as the average ratio of that image's percentiles to the means.

    Parameters
    ----------
    stack: A 4D array of images across all bits.
    quants: List of percentiles to use for normalization
    """
    percentiles = cp.array([cp.percentile(img[img > 0], quants) for img in stack])
    scale_factors = (percentiles.mean(axis=0) / percentiles).mean(axis=1)
    stack[:] = (stack.swapaxes(0, -1) * scale_factors).swapaxes(0, -1)


def highpass_filter(img: cp.ndarray, sigma: float = 3) -> None:
    """Apply a high-pass filter to the image.

    If the image is 3D, the high-pass filter is applied independently to each
    z-slice.

    Parameters
    ----------
    img: The image to apply the filter to
    ksize: Parameter passed to cv2.GaussianBlur. Numbers must be odd.
    sigma: Parameter passed to cv2.GaussianBlur

    Returns
    -------
    The filtered image
    """
    if img.ndim == 2:
        blur = gaussian_filter(img, sigma, mode="nearest")
        newimg = img - blur
        newimg[blur > img] = 0
        img[:] = newimg
    else:
        for i in range(0, img.shape[0]):
            highpass_filter(img[i], sigma)


def lowpass_filter(img: cp.ndarray, sigma: float = 1) -> None:
    """Apply a low-pass filter to the image."""
    if img.ndim == 2:
        img[:] = gaussian_filter(img, sigma, mode="nearest")
    else:
        for i in range(0, img.shape[0]):
            highpass_filter(img[i], sigma)


def flat_field_correct(img: cp.ndarray, sigma: float = 75) -> None:
    """Perform flat-field correction on an image.

    Flat-field correction is performed by dividing the image by a heavily blurred
    copy of the image, then re-scaling to maintain the original average brightness.
    If the image is 3D, the flat-field correction is performed independently on
    each z-slice.
    """
    if img.ndim == 2:
        dst = gaussian_filter(img, sigma, mode="nearest")
        avg_hist = img.mean()
        img[:] = (img / dst) * avg_hist
    else:
        for i in range(0, img.shape[0]):
            flat_field_correct(img[i], sigma)


def phase_cross_correlation(reference_image, moving_image, normalization="phase"):
    """Find translation between images using cross-correlation.

    This is adapted from the phase_cross_correlation function in skimage
    but uses the GPU for faster computation.
    """
    src_freq = fftn(reference_image)
    target_freq = fftn(moving_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = cp.finfo(image_product.real.dtype).eps
        image_product /= cp.maximum(cp.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = ifftn(image_product)

    # Locate maximum
    maxima = cp.unravel_index(
        cp.argmax(cp.abs(cross_correlation)), cross_correlation.shape
    )
    midpoints = cp.array([cp.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype

    shifts = cp.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= cp.array(shape)[shifts > midpoints]

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts


def calculate_drift(img1, img2, chunks=3):
    inds = np.linspace(0, img1.shape[0], num=chunks + 1, dtype=int)
    drifts = []
    for xstart, xstop in zip(inds, inds[1:]):
        for ystart, ystop in zip(inds, inds[1:]):
            drifts.append(
                phase_cross_correlation(
                    img1[xstart:xstop, ystart:ystop], img2[xstart:xstop, ystart:ystop]
                ).get()
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


def make_codebook_index(codebook_file):
    codebook = fileio.load_codebook(codebook_file)
    codes = codebook.filter(like="bit")
    X_codebook = np.ascontiguousarray(codes.to_numpy(), dtype=np.float32)
    faiss.normalize_L2(X_codebook)
    codebook_idx = faiss.IndexFlatL2(X_codebook.shape[1])
    try:
        res = faiss.StandardGpuResources()
        codebook_idx = faiss.index_cpu_to_gpu(res, 0, codebook_idx)
    except AttributeError:
        print("Faiss GPU not functional, using CPU")
    codebook_idx.add(np.ascontiguousarray(X_codebook, dtype=np.float32))
    return codebook_idx, X_codebook


def align_stack(stack, drifts):
    for i, (img, drift) in enumerate(zip(stack[2:], np.repeat(drifts, 2, axis=0))):
        stack[i + 2] = cp.roll(img, drift, axis=(-2, -1))


def load_fiducial_stack(data_dir, fov, zslice=0, rounds=11, channel=2):
    imgs = []
    for round in range(1, rounds + 1):
        path = os.path.join(data_dir, f"Conv_zscan_H{round}_F_{fov:03d}.dax")
        imgs.append(fileio.DaxFile(path, num_channels=3).zslice(zslice, channel))
    return cp.array(np.array(imgs))


def load_combinatorial_stack(data_dir, fov, zslice=None, rounds=11, bits_per_round=2):
    """Load all bit images for a single FOV.

    Parameters
    ----------
    data_dir: Directory containing the DAX files
    fov: The FOV number to load
    zslice: The zslice to load. If None, 3D images are loaded
    rounds: The number of hybridization rounds in the experiment
    bits_per_round: The number of colors used for bits per round

    Returns
    -------
    Numpy array of images
    """
    imgs = []
    for round in range(1, rounds + 1):
        path = os.path.join(data_dir, f"Conv_zscan_H{round}_F_{fov:03d}.dax")
        dax = fileio.DaxFile(path, num_channels=3)
        for bit in range(0, bits_per_round):
            if zslice:
                imgs.append(dax.zslice(zslice, bit))
            else:
                imgs.append(dax.channel(bit))
    return cp.array(np.array(imgs))


def assign_pixels_to_molecules(decoded, do_3d=False):
    def get_molecule_coords(id, z, x, y):
        if decoded[z, x, y] != id:
            return
        yield (z, x, y)
        decoded[z, x, y] = -1
        if x > 0:
            yield from get_molecule_coords(id, z, x - 1, y)
        if x < decoded.shape[1] - 1:
            yield from get_molecule_coords(id, z, x + 1, y)
        if y > 0:
            yield from get_molecule_coords(id, z, x, y - 1)
        if y < decoded.shape[2] - 1:
            yield from get_molecule_coords(id, z, x, y + 1)
        if do_3d:
            if z > 0:
                yield from get_molecule_coords(id, z - 1, x, y)
            if z < decoded.shape[0] - 1:
                yield from get_molecule_coords(id, z + 1, x, y)

    molecules = []
    mol_id = 0
    for z, x, y in zip(*np.where(decoded >= 0)):
        if decoded[z, x, y] >= 0:
            gene = decoded[z, x, y]
            molecules.extend(
                [
                    (mol_id, gene, *coords)
                    for coords in get_molecule_coords(gene, z, x, y)
                ]
            )
            mol_id += 1
    return cp.array(molecules)


def decode_pixels(stack, codebook_idx, X_codebook, distance_threshold=0.5167):
    X_pixels = cp.reshape(stack, (stack.shape[0], np.product(stack.shape[1:])))
    X_pixels = cp.moveaxis(X_pixels / cp.linalg.norm(X_pixels, axis=0), 0, -1).astype(
        np.float32
    )
    dists, indexes = codebook_idx.search(X_pixels.get(), k=1)

    decoded = cp.zeros(X_pixels.shape[0], dtype=cp.int32) - 1
    # faiss returns squared distance (to avoid sqrt op)
    valid = dists.flatten() <= distance_threshold ** 2
    decoded[valid] = indexes.flatten()[valid]
    decoded = cp.reshape(decoded, stack[0].shape)

    # molecules = assign_pixels_to_molecules(decoded)
    genes = cp.unique(decoded)[1:]
    labels = cp.zeros_like(decoded, dtype=cp.int32) - 1
    nlabels = 0
    for gene in genes:
        (gene_labels, gene_nlabels) = label(decoded == gene)
        mask = gene_labels != 0
        labels[mask] = gene_labels[mask] + nlabels
        nlabels += gene_nlabels
    nonzero = cp.where(decoded >= 0)
    molecules = cp.array(
        [labels[nonzero].flatten(), decoded[nonzero].flatten(), *cp.where(decoded >= 0)]
    ).T

    inds = cp.ravel_multi_index(cp.transpose(molecules[:, 2:]), stack.shape[1:])
    flatimg = cp.moveaxis(
        cp.reshape(stack, (stack.shape[0], np.product(stack.shape[1:]))), 0, -1
    )
    intensities = flatimg[inds]
    bits = intensities[X_codebook[molecules[:, 1].get()] > 0].reshape((inds.size, 4))
    mean_intensities = bits.mean(axis=1)
    snrs = mean_intensities / intensities.mean(axis=1)
    data = cp.array([mean_intensities, bits.max(axis=1), snrs]).T
    molecules = cp.concatenate([molecules, data, cp.array(dists[inds.get()])], axis=1)

    if stack.ndim == 4:
        columns = [
            "rna_id",
            "barcode_id",
            "z",
            "x",
            "y",
            "mean_intensity",
            "max_intensity",
            "snr",
            "distance",
        ]
    else:
        columns = [
            "rna_id",
            "barcode_id",
            "x",
            "y",
            "mean_intensity",
            "max_intensity",
            "snr",
            "distance",
        ]
    pixels = pd.DataFrame(molecules.get(), columns=columns)
    pixels["area"] = pixels.groupby("rna_id")["rna_id"].transform("count")
    return pixels


def combine_pixels_to_molecules(pixels):
    return pixels.groupby("rna_id").agg(
        {
            "z": "mean",
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


def decode_fov(folder, fov, codebook_idx, X_codebook):
    beads = load_fiducial_stack(folder, fov, zslice=0, rounds=11, channel=2)
    drifts = np.array([calculate_drift(beads[0], img) for img in beads[1:]])
    stack = load_combinatorial_stack(
        folder, fov, zslice=None, rounds=11, bits_per_round=2
    )
    flat_field_correct(stack)
    normalize_brightness(stack)
    highpass_filter(stack, sigma=2)
    # lowpass_filter(stack)
    align_stack(stack, drifts)
    pixels = decode_pixels(stack, codebook_idx, X_codebook)
    return combine_pixels_to_molecules(pixels)


def calc_misid(data):
    return (len(data[data["barcode_id"] < 10]) / 10) / (
        len(data[data["barcode_id"] >= 10]) / 238
    )


def filter_molecules_histogram(molecules):
    histogram = {
        "mean_intensity": np.percentile(molecules["mean_intensity"], range(1, 100)),
        "distance": np.percentile(molecules["distance"], range(1, 100)),
        "area": range(1, 26),
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
        if misid > 0.051:
            top = cutoff
        else:
            bottom = cutoff
        cutoff = (top + bottom) // 2
    return molecules[
        molecules["bin"].isin(bincounts.sort_values("ratio").head(cutoff).index)
    ]
