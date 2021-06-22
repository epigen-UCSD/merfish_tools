import os
from functools import cached_property

import pandas as pd
import numpy as np
from scipy.stats import zscore
from skimage.registration import phase_cross_correlation

import config


def announce(message: str):
    """
    This decorator is for functions that take some time to run, but can't use
    tqdm to show a progress bar. The message will be printed when the function
    starts running, then 'done' is printed when it finishes.
    """
    def decorator_announce(func):
        def announce_wrapper(*args, **kwargs):
            print(message+'...', end='', flush=True)
            rval = func(*args, **kwargs)
            print('done')
            return rval
        return announce_wrapper
    return decorator_announce


def csv_cached_property(csvname: str, save_index: bool = False, index_col: int = None):
    def decorator_csv(func):
        @cached_property
        def wrapper(self, *args, **kwargs):
            filename = os.path.join(self.analysis_folder, csvname)
            if not os.path.exists(filename) or config.get('rerun'):
                csv = func(self, *args, **kwargs)
                csv.to_csv(filename, index=save_index)
            return pd.read_csv(filename, index_col=index_col)
        return wrapper
    return decorator_csv


def expand_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    books = [codebook]
    for bit in range(1, 17):  # TODO: The number of bits shouldn't be hardcoded
        flip = codebook.copy()
        flip[f'bit{bit}'] = (~flip[f'bit{bit}'].astype(bool)).astype(int)
        flip['id'] = flip['id'] + f'_flip{bit}'
        books.append(flip)
    return pd.concat(books)


def calculate_drift(img1, img2):
    q1 = phase_cross_correlation(img1[:1024, :1024], img2[:1024, :1024])[0]
    q2 = phase_cross_correlation(img1[1024:, :1024], img2[1024:, :1024])[0]
    q3 = phase_cross_correlation(img1[1024:, 1024:], img2[1024:, 1024:])[0]
    q4 = phase_cross_correlation(img1[:1024, 1024:], img2[:1024, 1024:])[0]
    df = pd.DataFrame([q1, q2, q3, q4])
    if (df.std() > 15).any():
        df = df[(np.abs(zscore(df)) < 1.25).all(axis=1)]
    return df.median()
