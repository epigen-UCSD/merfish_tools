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
from cupyx.scipy.ndimage import gaussian_filter, label, shift, affine_transform
from cupyx.scipy.signal import convolve2d
from scipy import signal
from cupyx.scipy.fft import fftn, ifftn, fftfreq
from scipy.stats import zscore

from . import fileio

log = logging.getLogger(__name__)
rng = np.random.default_rng(seed=1047)


def gaussian_kernel(kernlen: int, std: float) -> np.ndarray:
    """Return a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


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
    log.info(f"Percentiles: {percentiles}")
    scale_factors = (percentiles.mean(axis=0) / percentiles).mean(axis=1)
    log.info(f"Scale factors: {scale_factors}")
    stack[:] = (stack.swapaxes(0, -1) * scale_factors).swapaxes(0, -1)


def highpass_filter(img: cp.ndarray, sigma: float = 3, truncate: int = 6) -> None:
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
        blur = gaussian_filter(img, sigma, truncate=truncate, mode="nearest")
        newimg = img - blur
        newimg[blur > img] = 0
        img[:] = newimg
    else:
        for i in range(img.shape[0]):
            highpass_filter(img[i], sigma)


def lowpass_filter(img: cp.ndarray, sigma: float = 1, truncate: int = 2) -> None:
    """Apply a low-pass filter to the image."""
    if img.ndim == 2:
        img[:] = gaussian_filter(img, sigma, truncate=truncate, mode="nearest")
    else:
        for i in range(img.shape[0]):
            lowpass_filter(img[i], sigma)


def flat_field_correct(img: cp.ndarray, sigma: float = 75, truncate: int = 150) -> None:
    """Perform flat-field correction on an image.

    Flat-field correction is performed by dividing the image by a heavily blurred
    copy of the image, then re-scaling to maintain the original average brightness.
    If the image is 3D, the flat-field correction is performed independently on
    each z-slice.
    """
    if img.ndim == 2:
        dst = gaussian_filter(img, sigma, truncate=truncate, mode="nearest")
        avg_hist = img.mean()
        img[:] = (img / dst) * avg_hist
    else:
        for i in range(img.shape[0]):
            flat_field_correct(img[i], sigma)


def calculate_projectors(window_size: int, sigma: float) -> list:
    """Calculate front and back projectors for deconvolution."""
    pf = cp.array(gaussian_kernel(window_size, sigma))
    pfFFT = cp.fft.fft2(pf)

    # These values are from Guo et al.
    alpha = 0.001
    beta = 0.001
    n = 8

    # This is the cut-off frequency
    kc = 1.0 / (0.5 * 2.355 * sigma)

    # FFT frequencies
    kv = cp.fft.fftfreq(pfFFT.shape[0])

    kx = cp.zeros((kv.size, kv.size))
    for i in range(kv.size):
        kx[i, :] = cp.copy(kv)

    ky = cp.transpose(kx)
    kk = cp.sqrt(kx * kx + ky * ky)

    # Wiener filter
    bWiener = pfFFT / (cp.abs(pfFFT) * cp.abs(pfFFT) + alpha)

    # Buttersworth filter
    eps = cp.sqrt(1.0 / beta ** 2 - 1)

    kkSqr = kk * kk / (kc * kc)
    bBWorth = 1.0 / cp.sqrt(1.0 + eps * eps * cp.power(kkSqr, n))

    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth

    # back projector.
    pb = cp.real(cp.fft.ifft2(pbFFT))

    return [pf, pb]


def deconvolve(img, window_size, sigma, iterations):
    """Perform Lucy-Richardson deconvolution.

    Adapted from the optimized implementation in MERlin:
    https://github.com/emanuega/MERlin/blob/master/merlin/util/deconvolve.py
    """
    if img.ndim == 2:
        # TODO: Calculate once and pass as parameter to all FOVs
        [pf, pb] = calculate_projectors(window_size, sigma)

        eps = 1.0e-6
        i_max = 2 ** 16 - 1

        ek = cp.copy(img.astype(float))
        cp.clip(ek, eps, None, ek)

        for _ in range(iterations):
            ekf = convolve2d(ek, pf, mode="same", boundary="symm")
            cp.clip(ekf, eps, i_max, ekf)

            ek = ek * convolve2d(img / ekf, pb, mode="same", boundary="symm")
            cp.clip(ek, eps, i_max, ek)

        img[:] = ek
    else:
        for i in range(img.shape[0]):
            deconvolve(img[i], window_size, sigma, iterations)


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    This is adapted from the function in skimage.registration
    but uses the GPU for faster computation.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    elif len(upsampled_region_size) != data.ndim:
        raise ValueError(
            "shape of upsampled region sizes must be equal "
            "to input data's number of dimensions."
        )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError(
            "number of axis offsets must be equal to input "
            "data's number of dimensions."
        )

    im2pi = 1j * 2 * cp.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = (cp.arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = cp.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = cp.tensordot(kernel, data, axes=(1, -1))
    return data


def phase_cross_correlation(
    reference_image, moving_image, upsample_factor=1, normalization="phase"
):
    """Find translation between images using cross-correlation.

    This is adapted from the function in skimage.registration
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

    shifts = cp.stack(maxima)  # .astype(np.float32, copy=False)
    shifts[shifts > midpoints] -= cp.array(shape)[shifts > midpoints]

    # If upsampling > 1, then refine estimate with matrix multiply DFT
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        upsample_factor = cp.array(upsample_factor)  # , dtype=np.float32)
        shifts = cp.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = float(np.ceil(upsample_factor * 1.5))
        # Center of output array at dftshift + 1
        dftshift = cp.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = cp.unravel_index(
            cp.argmax(cp.abs(cross_correlation)), cross_correlation.shape
        )

        maxima = cp.stack(maxima).astype(np.float32, copy=False)
        maxima -= dftshift

        shifts += maxima / upsample_factor

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    log.info(f"Drifts: {shifts}")
    return shifts


def calculate_drift_old(img1, img2, chunks=3, upsample_factor=1):
    """This didn't work well."""
    inds = np.linspace(0, img1.shape[0], num=chunks + 1, dtype=int)
    drifts = []
    for xstart, xstop in zip(inds, inds[1:]):
        for ystart, ystop in zip(inds, inds[1:]):
            drifts.append(
                phase_cross_correlation(
                    img1[xstart:xstop, ystart:ystop],
                    img2[xstart:xstop, ystart:ystop],
                    upsample_factor=upsample_factor,
                ).get()
            )
    df = pd.DataFrame(drifts)
    # Identify quadrants with outliers and remove
    if (df.std() > 15).any():
        df = df[(np.abs(zscore(df)) < 1.25).all(axis=1)]
    # If some quadrants are 0,0 they are probably bad, unless most are 0,0
    # then the non-zero ones are probably bad
    if df.any(axis=1).sum() > len(df) // 4:
        df = df[df.any(axis=1)]

    return np.array(df.mean())  # , dtype=np.int16)


def calculate_drift(img1, img2, upsample_factor=1):
    return phase_cross_correlation(img1, img2, upsample_factor=upsample_factor)


def get_drifts(folder, fov):
    beads = load_fiducial_stack(folder, fov, zslice=0, rounds=11, channel=2)
    return cp.array(
        [calculate_drift(beads[0], img, upsample_factor=100) for img in beads[1:]]
    )


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


def align_stack(stack, drifts, chromatic_corrector=None):
    if chromatic_corrector is not None:
        # from skimage.transform import warp

        for i in range(1, 23, 2):
            stack[i] = affine_transform(
                stack[i], chromatic_corrector, order=1, mode="nearest"
            )
            # stack[i] = cp.array(
            #    warp(stack[i].get(), chromatic_corrector, preserve_range=True)
            # )
    for i, (img, drift) in enumerate(zip(stack[2:], np.repeat(drifts, 2, axis=0))):
        if img.ndim == 3:
            drift = np.insert(drift.get(), 0, 0)
        stack[i + 2] = shift(img, drift, order=1)


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
        for bit in range(bits_per_round):
            if zslice:
                imgs.append(dax.zslice(zslice, bit))
            else:
                imgs.append(dax.channel(bit))
    return cp.array(np.array(imgs))


def decode_pixels(stack, codebook_idx, X_codebook, distance_threshold=0.5167):
    # print("Get neighbors")
    X_pixels = cp.reshape(stack, (stack.shape[0], np.product(stack.shape[1:])))
    X_pixels = cp.moveaxis(X_pixels / cp.linalg.norm(X_pixels, axis=0), 0, -1).astype(
        np.float32
    )
    dists, indexes = codebook_idx.search(X_pixels.get(), k=1)

    # print("Making decoded images")
    decoded = cp.zeros(X_pixels.shape[0], dtype=cp.int32) - 1
    # faiss returns squared distance (to avoid sqrt op)
    valid = dists.flatten() <= distance_threshold ** 2
    decoded[valid] = indexes.flatten()[valid]
    decoded = cp.reshape(decoded, stack[0].shape)

    # print("Assigning labels")
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

    # print("Getting pixel stats")
    inds = cp.ravel_multi_index(cp.transpose(molecules[:, 2:]), stack.shape[1:])
    flatimg = cp.moveaxis(
        cp.reshape(stack, (stack.shape[0], np.product(stack.shape[1:]))), 0, -1
    )
    intensities = flatimg[inds]
    bits = intensities[X_codebook[molecules[:, 1].get()] > 0].reshape((inds.size, 4))
    mean_intensities = bits.mean(axis=1)
    snrs = mean_intensities / intensities.mean(axis=1)
    data = cp.array([mean_intensities, bits.max(axis=1), snrs]).T
    data = cp.concatenate([intensities, data], axis=1)
    molecules = cp.concatenate([molecules, data, cp.array(dists[inds.get()])], axis=1)
    log.info(f"{len(molecules)} molecules decoded")

    # print("Constructing dataframe")
    columns = ["rna_id", "barcode_id"]
    if stack.ndim == 4:
        columns += ["z"]
    columns += ["x", "y"]
    columns += [f"bit{bit+1}" for bit in range(intensities.shape[1])]
    columns += ["mean_intensity", "max_intensity", "snr", "distance"]
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


def preprocess_stack(stack, drifts, scale_factors=None, chromatic_corrector=None):
    """Perform pre-processing on image stack."""
    align_stack(stack, drifts, chromatic_corrector)
    # flat_field_correct(stack)
    highpass_filter(stack)
    deconvolve(stack, 9, 2, 2)
    lowpass_filter(stack, sigma=1)
    if scale_factors is not None:
        stack[:] = (stack.swapaxes(0, -1) / scale_factors).swapaxes(0, -1)


def decode_fov(
    folder, fov, codebook_idx, X_codebook, scale_factors=None, chromatic_corrector=None
):
    drifts = get_drifts(folder, fov)
    stack = load_combinatorial_stack(
        folder, fov, zslice=None, rounds=11, bits_per_round=2
    )
    preprocess_stack(stack, drifts, scale_factors, chromatic_corrector)
    pixels = decode_pixels(stack, codebook_idx, X_codebook)
    molecules = combine_pixels_to_molecules(pixels)
    molecules["fov"] = fov
    return molecules


def decode_random_sample(
    folder,
    codebook_idx,
    X_codebook,
    sample_size,
    n_fovs,
    n_zstacks,
    scale_factors=None,
    chromatic_corrector=None,
):
    fovs = rng.integers(n_fovs, size=sample_size)
    zinds = rng.integers(n_zstacks, size=sample_size)
    mols = []
    pixs = []
    for fov, z in zip(fovs, zinds):
        # print(fov, z)
        beads = load_fiducial_stack(folder, fov, zslice=z, rounds=11, channel=2)
        drifts = cp.array(
            [calculate_drift(beads[0], img, upsample_factor=100) for img in beads[1:]]
        )
        stack = load_combinatorial_stack(
            folder, fov, zslice=z, rounds=11, bits_per_round=2
        )
        preprocess_stack(stack, drifts, scale_factors, chromatic_corrector)
        pix = decode_pixels(stack, codebook_idx, X_codebook)
        pix["z"] = z
        mols.append(combine_pixels_to_molecules(pix))
        pixs.append(pix)
    return pd.concat(pixs), pd.concat(mols)


def get_scale_factors(pixels, X_codebook, area=5):
    keep = pixels[pixels["area"] >= area].groupby("rna_id").mean()
    ints = keep.filter(like="bit").to_numpy()
    ints[X_codebook[keep["barcode_id"].astype(int)] == 0] = np.nan
    scale_factors = np.nanmean(ints, axis=0)
    return np.nan_to_num(scale_factors / np.nanmean(scale_factors), nan=1)


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
