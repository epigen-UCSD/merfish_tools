import argparse
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


def print_hamming_dists(codebook):
    pdist_ = pdist(codebook, metric="hamming") * len(codebook[0])
    ham_dists, cts = np.unique(pdist_, return_counts=True)
    print("Hamming distances and counts:")
    print(ham_dists)
    print(cts)


def get_initial_barcodes(number_of_bits):
    web = 'http://ljcr.dmgordon.org/cover/show_cover.php?v='+str(number_of_bits)+'&k=4&t=3'
    print(web)
    with urllib.request.urlopen(web) as f:
        txt = f.read().decode('utf-8')
    elems = txt.split('>')
    txt_ = elems[np.argmax([len(e) for e in elems])].split('<')[0]
    code = np.array([[e for e in elem.split(' ') if len(e)] for elem in txt_.split('\n') if len(elem)], dtype=int) - 1

    bin_code = np.zeros([len(code), np.max(code) + 1], dtype=int)
    for icd, cd in enumerate(code):
        bin_code[icd][cd] = 1

    return bin_code

def cleanup_barcodes(bin_code):
    # For some codes the hamming distance >= 4 is not always satisfied.
    # Cleanup by throwing away codes untill that is sattisfied
    bin_code_ = list(bin_code)
    pdist_ = pdist(bin_code_, metric="hamming") * len(bin_code_[0])
    M = squareform(pdist_)
    M[range(len(M)), range(len(M))] = np.inf
    bad_pairs = np.array(np.where(M <= 2)).T
    bad_inds, ct_inds = np.unique(bad_pairs, return_counts=True)
    bad_inds_save = list(bad_inds)
    final_remove = []
    if len(bad_inds) > 0:
        bin_code__ = list(bin_code[bad_inds])
        while True:
            pdist_ = pdist(bin_code__, metric="hamming") * len(bin_code__[0])
            M = squareform(pdist_)
            M[range(len(M)), range(len(M))] = np.inf
            bad_pairs = np.array(np.where(M <= 2)).T
            bad_inds, ct_inds = np.unique(bad_pairs, return_counts=True)
            if len(bad_inds) > 0:
                badI = bad_inds[np.argmax(ct_inds)]
                bin_code__.pop(badI)
                final_remove.append(bad_inds_save[badI])
                bad_inds_save.pop(badI)
            else:
                break

    return bin_code[np.setdiff1d(np.arange(len(bin_code)), final_remove)]


def best_spaced(nlen, l):
    "In the range 0 to nlen return the ~equally spaced l integers"
    return list(np.linspace(0, nlen - 1, l).astype(int))


def metric(bin_code, keep):
    """Metric for what the MH algortith should optimize for.
    This tries to make the number of genes across the bits as uniform as possible"""
    return np.std(np.sum(bin_code[keep], 0))


def umi_metric(final_code_genes, keep, umis):
    """Metric for what the MH algortith should optimize for.
    This tries to make the number of genes across the bits as uniform as possible"""
    return np.std(np.dot(umis.loc[keep][1], final_code_genes))


def randomchange(list1, list2=None):
    """Randomly switch 2 elements from two lists."""
    list1_ = list1[:]
    if list2 is None:
        list2_ = list1_
    else:
        list2_ = list2[:]
    i1 = np.random.randint(len(list1_))
    i2 = np.random.randint(len(list2_))
    list1_[i1], list2_[i2] = list2_[i2], list1_[i1]
    if list2 is None:
        return list1_
    else:
        return list1_, list2_


def balance_codes(bin_code, keep, leftover):
    """Balances the bits used in keep by swapping barcodes between keep and leftover"""
    m0 = np.inf
    m_abs = np.inf
    beta = 1000  # temperature paramater

    ms = []
    ms_ = []
    for irep in tqdm(range(30000)):
        keep_, leftover_ = randomchange(keep, leftover)  # switch two elements
        m_ = metric(bin_code, keep_)  # evaluate how uniform
        if m_ < m_abs:
            keep, leftover = keep_, leftover_
            m0 = m_
            m_abs = m_
            keep_final, leftover_final = keep, leftover
            ms.append(m0)
        else:
            p = np.exp((m0 - m_) * beta)
            if np.random.rand() < p:
                keep, leftover = keep_, leftover_
                m0 = m_
        ms_.append(m_)

    return keep_final, leftover_final

def balance_umis(keep, umis, final_code_genes):
    """Balances the bits used in keep by swapping barcodes between keep and leftover"""
    m0 = np.inf
    m_abs = np.inf
    beta = 1000  # temperature paramater

    ms = []
    ms_ = []
    for irep in tqdm(range(30000)):
        keep_ = randomchange(keep)  # switch two elements
        m_ = umi_metric(final_code_genes, keep_, umis)  # evaluate how uniform
        if m_ < m_abs:
            keep = keep_
            m0 = m_
            m_abs = m_
            keep_final = keep
            ms.append(m0)
        else:
            p = np.exp((m0 - m_) * beta)
            if np.random.rand() < p:
                keep = keep_
                m0 = m_
        ms_.append(m_)

    return keep_final


def create_codebook_template(number_of_bits, genes, blanks):
    barcodes = get_initial_barcodes(number_of_bits)

    print("\nInitial barcodes")
    print("Number of codes, length of code:", len(barcodes), len(barcodes[0]))
    print_hamming_dists(barcodes)

    barcodes = cleanup_barcodes(barcodes)

    print("\nCleaned-up barcodes")
    print("Number of codes, length of code:", len(barcodes), len(barcodes[0]))
    print_hamming_dists(barcodes)

    if genes + blanks > len(barcodes):
        print("\nERROR: Impossible codebook configuration!")
        print(f"There are a total of {len(barcodes)} {number_of_bits}-bit barcodes,")
        print(f"but {genes+blanks} have been requested ({genes} genes, {blanks} blanks)")
        quit()
    elif genes + blanks == len(barcodes):
        # Must use all possible barcodes
        keep = list(range(len(barcodes)))
        print("\nUsing all barcodes (genes + blanks is equal to total)")
    else:
        # Initialize with equally spaced codes
        initial = best_spaced(len(barcodes), genes + blanks)  # indices of the initial selection
        leftover = list(np.setdiff1d(np.arange(len(barcodes)), initial))  # indices of the leftover barcodes
        keep, _ = balance_codes(barcodes, initial, leftover)
        print(f"\nSelected {len(barcodes[keep])} barcodes")

    print("Number of barcodes per bit:")
    print(np.sum(barcodes[keep], 0))

    initial_blanks = keep[:blanks]
    initial_genes = keep[blanks:]

    blanks, genes = balance_codes(barcodes, initial_blanks, initial_genes)

    blank_barcodes = barcodes[blanks]
    gene_barcodes = barcodes[genes]

    print(f"\nSelected {len(blank_barcodes)} blank barcodes")
    print("Number of barcodes per bit:")
    print(np.sum(blank_barcodes, 0))
    print_hamming_dists(blank_barcodes)

    print(f"\nSelected {len(gene_barcodes)} gene barcodes")
    print("Number of genes per bit:")
    print(np.sum(gene_barcodes, 0))
    print_hamming_dists(gene_barcodes)

    return blank_barcodes, gene_barcodes


def main(args):
    if args.load_template is None:
        blank_barcodes, gene_barcodes = create_codebook_template(args.number_of_bits, arg.genes, args.blanks)
        if args.save_template is not None:
            with open(args.save_template, 'w') as f:
                for num, barcode in enumerate(blank_barcodes, start=1):
                    print(f'blank{num:04}' + ',' + ','.join(barcode.astype(str)), file=f)
                for num, barcode in enumerate(gene_barcodes, start=1):
                    print(f'gene{num:05}' + ',' + ','.join(barcode.astype(str)), file=f)
    else:
        blank_barcodes = []
        gene_barcodes = []
        with open(args.load_template) as f:
            for line in f:
                if line.startswith('blank'):
                    blank_barcodes.append(line.strip().split(',')[1:])
                elif line.startswith('gene'):
                    gene_barcodes.append(line.strip().split(',')[1:])
        blank_barcodes = np.array(blank_barcodes).astype(int)
        gene_barcodes = np.array(gene_barcodes).astype(int)

    if args.umis_file is not None:
        umis = pd.read_csv(args.umis_file, header=None, index_col=0)
        order = list(umis.index)
        order = balance_umis(order, umis, gene_barcodes)
        print(np.dot(umis.loc[order][1], gene_barcodes))
        with open(args.output_file, 'w') as f:
            print("name,id,bit1,bit2,bit3,bit4,bit5,bit6,bit7,bit8,bit9,bit10,bit11,bit12,bit13,bit14,bit15,bit16", file=f)
            for num, barcode in enumerate(blank_barcodes, start=1):
                print(f'blank{num:04}' + ',' + f'blank{num:04}' + ',' + ','.join(barcode.astype(str)), file=f)
            for gene, barcode in zip(order, gene_barcodes):
                print(f'{gene}' + ',' + f'{gene}' + ',' + ','.join(barcode.astype(str)), file=f)


def parse_args():
    parser = argparse.ArgumentParser(description='Design an empty MERFISH codebook.')
    parser.add_argument('-l', '--length', help='The length in bits of the barcodes',
                        dest='number_of_bits', type=int)
    parser.add_argument('-b', '--blanks', help='The number of blank barcodes to include',
                        dest='blanks', type=int)
    parser.add_argument('-g', '--genes', help='The number of gene barcodes to include',
                        dest='genes', type=int)
    parser.add_argument('-s', '--save_template', help='Path to save the codebook template',
                        dest='save_template', type=str)
    parser.add_argument('-t', '--load_template', help='Load existing codebook template instead of creating new one',
                        dest='load_template', type=str)
    parser.add_argument('-u', '--umis', help='CSV table of genes to assign with UMIs to balance',
                        dest='umis_file', type=str)
    parser.add_argument('-o', '--output', help='Filename to save codebook',
                        dest='output_file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
