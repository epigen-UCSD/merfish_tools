import argparse
import os

from mftools import config
from mftools import fileio
from mftools import segmentation
from mftools import util
from mftools import stats
from mftools import barcodes
from mftools import cellgene
from mftools import plotting


def create_barcode_table(merlin_result, masks, cell_links):
    codebook = merlin_result.load_codebook()
    bcs = barcodes.make_table(merlin_result, codebook)
    per_bit_error = barcodes.set_barcode_stats(merlin_result, bcs, config.get("barcode_colors"))
    plotting.exact_vs_corrected()
    per_gene_error = barcodes.per_gene_error(bcs)
    plotting.confidence_ratios(per_gene_error)
    if per_bit_error is not None:
        plotting.per_bit_error_bar(per_bit_error, config.get("barcode_colors"))
        plotting.per_bit_error_line(per_bit_error, config.get("barcode_colors"))
        plotting.per_hyb_error(per_bit_error)
        plotting.per_color_error(per_bit_error, config.get("barcode_colors"))
    # per_fov_error = barcodes.per_fov_error(bcs)
    # plotting.fov_error_bar(per_fov_error)
    # plotting.fov_error_spatial(per_fov_error, positions)
    # plotting.spatial_transcripts_per_fov(bcs, positions)
    barcodes.mark_barcodes_in_overlaps(bcs, masks.positions.find_fov_overlaps(get_trim=True))
    barcodes.assign_to_cells(bcs, masks)
    barcodes.calculate_global_coordinates(
        bcs, masks.positions.positions
    )  # Replace with util.fov_to_global_coordinates
    barcodes.link_cell_ids(bcs, cell_links)
    for dataset in config.get("reference_counts"):
        plotting.rnaseq_correlation(bcs, dataset)
    return bcs


def analyze_experiment():
    stats.savefile = config.path("stats.json")
    merlin_result = fileio.MerlinOutput(config.get("merlin_folder"))
    if config.has("image_folder"):
        imagedata = fileio.ImageDataset(
            config.get("image_folder"),
            data_organization=merlin_result.load_data_organization(),
            segdict={"hyb": 0, "frame": 53},
        )
    output = fileio.MerfishAnalysis(config.get("output_folder"))
    masks = segmentation.CellSegmentation(
        config.get("segmentation_folder"),
        output=output,
        positions=merlin_result.load_fov_positions(),
        imagedata=imagedata,
    )

    positions = merlin_result.load_fov_positions()
    n_fovs = len(positions)
    stats.set("FOVs", n_fovs)

    # f os.path.exists(config.path("cell_metadata.csv")):
    # celldata = fileio.load_cell_metadata(config.path("cell_metadata.csv"))
    # cell_links = fileio.load_cell_links(config.path("cell_links.txt"))
    # else:
    celldata = masks.metadata
    cell_links = masks.linked_cells
    stats.set("Segmented cells", len(celldata))

    bcs = create_barcode_table(merlin_result, masks, cell_links)
    fileio.save_barcode_table(bcs, config.path("barcodes.csv"))

    counts = barcodes.create_cell_by_gene_table(bcs)
    fileio.save_cell_by_gene_table(counts, config.path("cell_by_gene.csv"))
    plotting.counts_per_cell_histogram(counts)
    plotting.genes_detected_per_cell_histogram(counts)
    codebook = merlin_result.load_codebook()
    adata = cellgene.create_scanpy_object(counts, celldata, positions, codebook)
    adata.write(config.path("scanpy_object.h5ad"))
    cellgene.normalize(adata)
    n_pcs = cellgene.optimize_number_PCs(adata)
    cellgene.cluster_cells(adata, n_pcs)
    adata.write(config.path("scanpy_object.h5ad"))
    plotting.umap_clusters(adata)
    plotting.spatial_cell_clusters(adata)


def main():
    parser = argparse.ArgumentParser(description="Run the MERFISH analysis pipeline.")
    parser.add_argument(
        "-e",
        "--experiment",
        help="The name of the experiment",
        dest="experiment_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--analysis_root",
        help="Location of MERlin analysis directories",
        dest="analysis_root",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--data_root",
        help="Location of MERlin raw data folders",
        dest="data_root",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--rerun",
        help="Force rerun all steps, overwriting existing files",
        dest="rerun",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to the configuration file in JSON format",
        dest="config_file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Sub-folder to save all files produced by the pipeline",
        dest="result_folder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    config.load(args)
    analyze_experiment()


if __name__ == "__main__":
    main()
