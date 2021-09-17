import argparse

import config, plotting
from experiment import MerfishExperiment


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
    mfx = MerfishExperiment()
    mfx.stats.calculate_decoding_metrics()
    # print(mfx.stats['Median transcripts per cell'])
    # print(mfx.stats['Median genes detected per cell'])
    # for dataset in config.get('reference_counts'):
    #        plotting.rnaseq_correlation(mfx, dataset['name'])


if __name__ == "__main__":
    main()
