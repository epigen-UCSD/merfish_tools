"""Calculates and stores various statistics and quality metrics.

This module should generally not be interacted with directly, but rather through an instance of
the MerfishExperiment class (see experiment.py).
"""

import os
import json
import atexit
import random
from functools import cached_property

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.linalg import norm
from scipy.stats import pearsonr

import config
import plotting
from util import announce, csv_cached_property


class Stats:
    """This class calculates and stores various statistics about MERFISH experiments."""

    def __init__(self, mfx):
        """Create Stats object and load existing stats from file, if it exists. mfx is a MerfishExperiment."""
        self.mfx = mfx
        self.analysis_folder = self.mfx.analysis_folder
        self.filepath = config.path('stats.json')
        self.functions = {
            'FOVs': lambda stats: len(mfx.fovs),
            'Unfiltered barcode count': count_unfiltered_barcodes,
            'Filtered barcode count': lambda stats: len(stats.mfx.unassigned_barcodes),
            'Unfiltered barcodes per FOV': lambda stats: stats['Unfiltered barcode count'] / stats['FOVs'],
            'Filtered barcodes per FOV': lambda stats: stats['Filtered barcode count'] / stats['FOVs'],
            '% barcodes kept': lambda stats: stats['Filtered barcode count'] / stats['Unfiltered barcode count'],
            'Exact barcode count': lambda stats: len(self.mfx.unassigned_barcodes[self.mfx.unassigned_barcodes['error_type'] == 0]),
            'Corrected barcode count': lambda stats: stats['Filtered barcode count'] - stats['Exact barcode count'],
            '% exact barcodes': lambda stats: stats['Exact barcode count'] / stats['Filtered barcode count'],
            '0->1 error rate': lambda stats: len(self.mfx.unassigned_barcodes[self.mfx.unassigned_barcodes['error_type'] == 1]) / stats['Filtered barcode count'],
            '1->0 error rate': lambda stats: len(self.mfx.unassigned_barcodes[self.mfx.unassigned_barcodes['error_type'] == -1]) / stats['Filtered barcode count'],
            #'Pre-filtering 0->1 error rate': lambda stats: self.global_error_prefiltered['0->1 error rate'],
            #'Pre-filtering 1->0 error rate': lambda stats: self.global_error_prefiltered['1->0 error rate'],
            'Average per-bit 0->1 error rate': lambda stats: self.per_bit_error[self.per_bit_error['Error type'] == '0->1']['Error rate'].mean(),
            'Average per-bit 1->0 error rate': lambda stats: self.per_bit_error[self.per_bit_error['Error type'] == '1->0']['Error rate'].mean(),
            'Segmented cells': lambda stats: len(np.unique(self.mfx.celldata.index)),
            'Segmented cells per FOV': lambda stats: stats['Segmented cells'] / stats['FOVs'],
            'Median cell volume (pixels)': lambda stats: np.median(stats.mfx.celldata['volume']),
            'Barcodes assigned to cells': lambda stats: len(self.mfx.assigned_barcodes[self.mfx.assigned_barcodes['status'] == 'good']),
            '% barcodes assigned to cells': lambda stats: stats['Barcodes assigned to cells'] / len(self.mfx.assigned_barcodes[self.mfx.assigned_barcodes['status'] != 'edge']),
            'Cells with barcodes': lambda stats: len(np.unique(self.mfx.assigned_barcodes[self.mfx.assigned_barcodes['status'] == 'good']['cell_id'])),
            '% cells with barcodes': lambda stats: stats['Cells with barcodes'] / stats['Segmented cells'],
            'Median transcripts per cell': lambda stats: np.median(self.mfx.single_cell_raw_counts.apply(np.sum, axis=1)),
            'Median genes detected per cell': lambda stats: np.median(self.mfx.single_cell_raw_counts.astype(bool).sum(axis=1)),
            'PCs used in clustering': lambda stats: self.mfx.clustering.number_PCs,
            'Number of clusters': lambda stats: self.mfx.clustering.nclusts
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
        """Calculate the stat if not available, then return it."""
        if stat not in self.stats:
            self.stats[stat] = self.functions[stat](self)
            if not self.unsaved:
                self.unsaved = True
                atexit.register(save_stats, stats=self)
        return self.stats[stat]

    @csv_cached_property('error_counts_prefiltered.csv')
    def error_counts_prefiltered(self) -> pd.DataFrame:
        """Calculate the barcode error statistics before adaptive filtering was applied.

        The FOVs are processed one at a time in a random order and stopped when the overall
        error rate changes by less than 0.1% when adding a new FOV. There are so many barcodes
        before filtering that processing all FOVs would take a very long time.
        """
        codebook = self.mfx.expanded_codebook
        codes = codebook.filter(like='bit')
        normcodes = codes.apply(lambda row: row / norm(row), axis=1)

        files = random.sample(self.mfx.raw_barcode_files, self['FOVs'])

        data = None
        zero2one = 100
        one2zero = 100
        stable_count = 0
        for filename in tqdm(files, desc='Sampling pre-filtering barcodes', total=float("inf")):
            newdata = process_fov(filename, normcodes, codebook)
            data = pd.concat([data, newdata], ignore_index=True)
            stats = error_stats(data)
            if abs(stats['0->1 error rate'] - zero2one) < 0.001 and abs(stats['1->0 error rate'] - one2zero) < 0.001:
                stable_count += 1
                print(stable_count)
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
            zero2one = stats['0->1 error rate']
            one2zero = stats['1->0 error rate']

        return data


    @csv_cached_property('per_gene_error.csv', save_index=True)
    def per_gene_error(self) -> pd.DataFrame:
        """Get barcode error statistics per gene."""
        temp = self.mfx.unassigned_barcodes.groupby(['gene', 'error_type']).count().reset_index().pivot(index='gene', columns='error_type', values='barcode_id')
        temp.columns = ['1->0 errors', 'Exact barcodes', '0->1 errors']
        temp['Total barcodes'] = temp['1->0 errors'] + temp['0->1 errors'] + temp['Exact barcodes']
        temp['% exact barcodes'] = temp['Exact barcodes'] / temp['Total barcodes']
        return temp

    @cached_property
    def per_bit_error(self) -> pd.DataFrame:
        """Get barcode error statistics per bit."""
        err_rates = pd.concat(get_per_bit_stats(gene, group) for gene, group in self.mfx.unassigned_barcodes.groupby('gene'))
        err_rates['Color'] = err_rates.apply(lambda row: self.mfx.barcode_colors[(row['Bit']-1) % len(self.mfx.barcode_colors)], axis=1)
        err_rates['Hybridization round'] = err_rates.apply(lambda row: ((row['Bit']-1) // len(self.mfx.barcode_colors))+1, axis=1)
        return err_rates

    @cached_property
    def per_fov_error(self) -> pd.DataFrame:
        """Get barcode error statistics per FOV."""
        foverr = self.mfx.unassigned_barcodes.groupby(['fov', 'error_type']).count()['gene'] / self.mfx.unassigned_barcodes.groupby('fov').count()['gene']
        foverr = foverr.reset_index()
        foverr.columns = ['FOV', 'Error type', 'Error rate']
        foverr['Error type'] = foverr['Error type'].replace([0, -1, 1], ['Correct', '1->0', '0->1'])
        return foverr

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
            refcounts = pd.read_csv('/storage/RNA_MERFISH/Reference_Data/W15_all_genes_norm.csv')
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
        plotting.fov_number_map(self.mfx)
        plotting.fov_error_bar(self.mfx)
        plotting.fov_error_spatial(self.mfx)
        for dataset in config.get('reference_counts'):
            plotting.rnaseq_correlation(self.mfx, dataset['name'])
        plotting.cell_volume_histogram(self.mfx)
        plotting.counts_per_cell_histogram(self.mfx)
        plotting.genes_detected_per_cell_histogram(self.mfx)
        plotting.spatial_cell_clusters(self.mfx)

    @announce("Saving statistics")
    def save(self) -> None:
        text = json.dumps(self.stats, indent=4)
        with open(self.filepath, 'w') as f:
            f.write(text)
        atexit.unregister(save_stats)
        self.unsaved = False


def save_stats(stats: Stats) -> None:
    """Call the save method of the given Stats object.

    This function is needed to get the atexit.register stuff to work, as you can't use it directly
    with class methods. Whenever something is added or changed in a Stats object, we use the atexit
    library to register this save function to run when python exits. This lets it get saved even
    if the program hits an error and crashes.
    """
    stats.save()


def count_unfiltered_barcodes(stats: Stats) -> int:
    """Count the total number of barcodes decoded by MERlin before adaptive filtering."""
    raw_count = 0
    for file in tqdm(stats.mfx.raw_barcode_files, desc="Counting unfiltered barcodes"):
        barcodes = pd.read_hdf(file)
        raw_count += len(barcodes)
    return raw_count


def get_per_bit_stats(gene: str, barcodes: pd.DataFrame) -> pd.DataFrame:
    k0 = len(barcodes[barcodes['error_type'] == 0])
    total = len(barcodes)
    rows = []
    for bit in range(1, 23):
        errors = barcodes[barcodes['error_bit'] == bit]
        k1 = len(errors)
        err_type = '1->0' if errors.iloc[0]['error_type'] == -1 else '0->1'
        rate = (k1 / k0) / (1 + (k1 / k0))
        rows.append([gene, bit, k1, err_type, rate, rate*total, total])
    return pd.DataFrame(rows, columns=['gene', 'Bit', 'count', 'Error type', 'Error rate', 'weighted', 'total']).reset_index()
