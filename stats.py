"""Calculates and stores various statistics and quality metrics.

This module should generally not be interacted with directly, but rather through an instance of
the MerfishExperiment class (see experiment.py).
"""

import os
import glob
import json
import atexit
from functools import cached_property
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.linalg import norm
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

import config
import plotting
from util import announce, csv_cached_property


class Stats:
    """
    This class stores and calculates various statistics about MERFISH experiments.
    """
    def __init__(self, mfx: 'MerfishExperiment'):
        self.mfx = mfx
        self.analysis_folder = self.mfx.analysis_folder
        self.filepath = os.path.join(mfx.analysis_folder, 'stats.json')
        self.functions = {
            'FOVs': lambda stats: len(stats.mfx.raw_barcode_files),
            'Unfiltered barcode count': count_unfiltered_barcodes,
            'Filtered barcode count': lambda stats: len(stats.mfx.barcodes),
            'Unfiltered barcodes per FOV': lambda stats: stats['Unfiltered barcode count'] / stats['FOVs'],
            'Filtered barcodes per FOV': lambda stats: stats['Filtered barcode count'] / stats['FOVs'],
            '% barcodes kept': lambda stats: stats['Filtered barcode count'] / stats['Unfiltered barcode count'],
            'Exact barcode count': lambda stats: self.global_error['Exact barcode count'],
            'Corrected barcode count': lambda stats: self.global_error['Corrected barcode count'],
            '% exact barcodes': lambda stats: self.global_error['% exact barcodes'],
            '0->1 error rate': lambda stats: self.global_error['0->1 error rate'],
            '1->0 error rate': lambda stats: self.global_error['1->0 error rate'],
            'Average per-bit 0->1 error rate': lambda stats: self.per_bit_error[self.per_bit_error['Error type'] == '1->0']['Error rate'].mean(),
            'Average per-bit 1->0 error rate': lambda stats: self.per_bit_error[self.per_bit_error['Error type'] == '0->1']['Error rate'].mean()
        }
        if config.has('reference_counts'):
            for dataset in config.get('reference_counts'):
                self.functions[dataset['name']] = lambda stats: self.correlation_with_reference(dataset['file'])
        if not config.get('rerun') and os.path.exists(self.filepath):
            self.stats = json.load(open(self.filepath))
        else:
            self.stats = {}
        self.unsaved = False
        self._ref_counts = {}

    def __getitem__(self, stat: str):
        if stat not in self.stats:
            self.stats[stat] = self.functions[stat](self)
            if not self.unsaved:
                self.unsaved = True
                atexit.register(save_stats, stats=self)
        return self.stats[stat]

    @csv_cached_property('error_counts.csv')
    def error_counts(self) -> pd.DataFrame:
        #L2 normalize the expanded codebook
        codebook = self.mfx.expanded_codebook
        codes = codebook.filter(like='bit')
        normcodes = codes.apply(lambda row: row / norm(row), axis=1)

        dfs = [process_fov(filename, normcodes, codebook) for filename in tqdm(self.mfx.filtered_barcode_files, desc='Getting error rates')]
        return pd.concat(dfs, ignore_index=True)

    @cached_property
    def global_error(self) -> pd.Series:
        return error_stats(self.error_counts)

    @csv_cached_property('per_gene_error.csv', save_index=True, index_col=0)
    def per_gene_error(self) -> pd.DataFrame:
        return self.error_counts.groupby('name').apply(error_stats).sort_values(by='% exact barcodes', ascending=False)

    @cached_property
    def per_bit_error(self) -> pd.DataFrame:
        err_rates = self.error_counts.groupby('name').apply(get_per_bit_stats).reset_index()
        err_rates['Color'] = err_rates.apply(lambda row: self.mfx.barcode_colors[(row['Bit']-1) % len(self.mfx.barcode_colors)], axis=1)
        err_rates['Hybridization round'] = err_rates.apply(lambda row: ((row['Bit']-1) // len(self.mfx.barcode_colors))+1, axis=1)
        return err_rates

    @cached_property
    def per_fov_error(self) -> pd.DataFrame:
        fovs = self.error_counts.groupby('fov').apply(error_stats)
        fovs['FOV'] = fovs.index
        fovs.columns = ['Count', 'No errors', 'Errors', 'Correct', '0 -> 1', '1 -> 0', 'FOV']
        fovs = fovs.melt(id_vars='FOV', value_vars=['0 -> 1', '1 -> 0'], var_name='Error type', value_name='Error rate')
        return fovs

    @cached_property
    def merfish_gene_counts(self) -> dict:
        gene_counts = np.load(os.path.join(self.analysis_folder, "PlotPerformance", "filterplots", "FilteredBarcodesMetadata", "barcode_counts.npy"))
        ordered_genes = list(self.mfx.codebook['name'])
        abundances = {}
        for i, gene in enumerate(ordered_genes):
            abundances[gene] = np.log10(gene_counts[i]+1)
        return abundances

    def reference_gene_counts(self, filename) -> dict:
        if filename not in self._ref_counts:
            refcounts = pd.read_csv('/storage/RNA_MERFISH/Reference_Data/W15_scrna_norm_counts.csv')
            self._ref_counts[filename] = dict(zip(refcounts['geneName'], np.log10(refcounts['counts'])))
        return self._ref_counts[filename]

    def correlation_with_reference(self, filename, omit=[]) -> float:
        xcounts = self.merfish_gene_counts
        ycounts = self.reference_gene_counts(filename)
        set1 = set(xcounts.keys())
        set2 = set(ycounts.keys())
        genes_to_consider = [gene for gene in list(set1.intersection(set2)) if gene not in omit]
        x = [xcounts[gene] for gene in genes_to_consider]
        y = [ycounts[gene] for gene in genes_to_consider]
        corr, pval = pearsonr(x, y)
        return corr

    def calculate_decoding_metrics(self) -> None:
        for stat in self.functions:
            self[stat]
        plotting.drift_histogram(self.mfx)
        plotting.exact_vs_corrected(self.mfx)
        plotting.confidence_ratios(self.mfx)
        plotting.per_bit_error_bar(self.mfx)
        plotting.per_bit_error_line(self.mfx)
        plotting.per_color_error(self.mfx)
        plotting.per_hyb_error(self.mfx)
        plotting.fov_error_bar(self.mfx)
        plotting.fov_error_spatial(self.mfx)
        for dataset in config.get('reference_counts'):
            plotting.rnaseq_correlation(self.mfx, dataset['name'])

    @announce("Saving statistics")
    def save(self) -> None:
        with open(self.filepath, 'w') as f:
            f.write(json.dumps(self.stats, indent=4))
        atexit.unregister(save_stats)
        self.unsaved = False


def save_stats(stats: Stats) -> None:
    stats.save()

def count_unfiltered_barcodes(stats: Stats) -> int:
    raw_count = 0
    for file in tqdm(stats.mfx.raw_barcode_files, desc="Counting unfiltered barcodes"):
        barcodes = pd.read_hdf(file)
        raw_count += len(barcodes)
    return raw_count

def process_fov(filename: str, normcodes: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    barcodes = pd.read_hdf(filename)
    fov = int(filename.split('_')[-1].split('.')[0])

    #Get just the intensity columns for convenience
    intensities = barcodes[[f'intensity_{i}' for i in range(16)]]
    intensities = intensities.rename(columns=lambda x: 'intensity_' + str(int(x.split('_')[1])+1))

    #Find nearest
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=16)
    neighbors.fit(normcodes)
    distances, indexes = neighbors.kneighbors(intensities, return_distance=True)

    df = codebook.iloc[indexes.T[0]].copy()
    df['fov'] = fov
    df = df.set_index(['name', 'id', 'fov']).filter(like='bit').sum(axis=1)
    df = pd.DataFrame(df, columns=['bits']).set_index(['bits'], append=True)
    df['count'] = df.index.value_counts()

    return df.reset_index().drop_duplicates()

def error_stats(data: pd.DataFrame) -> pd.Series:
    c = data.groupby('bits')['count'].sum()
    total = sum(c)
    exact = c[4] if 4 in c else 0
    one2zero = c[3] if 3 in c else 0
    zero2one = c[5] if 5 in c else 0
    columns = ["Barcodes", "Exact barcode count", "Corrected barcode count", "% exact barcodes", "0->1 error rate", "1->0 error rate"]
    return pd.Series([total, exact, one2zero+zero2one, exact / total, zero2one / total, one2zero / total], index=columns)

def get_per_bit_stats(data: pd.DataFrame) -> pd.DataFrame:
    total = data['count'].sum()
    data = data[data['id'] != data.name]
    err_rates = pd.DataFrame(data.groupby(['id', 'bits'])['count'].sum() / total).reset_index()
    err_rates = err_rates.replace({'bits': {5: '0->1', 3: '1->0'}})
    err_rates = err_rates.replace({'id': r'[^_]+_flip'}, {'id': ''}, regex=True)
    err_rates.columns = ['Bit', 'Error type', 'Error rate']
    err_rates['Bit'] = pd.to_numeric(err_rates['Bit'])

    return err_rates.set_index('Bit')
