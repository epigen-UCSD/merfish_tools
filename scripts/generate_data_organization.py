"""Data organization file generator.

This script generates the data organization file needed by MERlin. You must
specify 3 command-line parameters:

  -b/--bits
    The number of bits used for MERFISH encoding.

  -c/--colors
    The colors used for MERFISH encoding. Do not include fiducial color channel.
    For example, if 750 and Cy5 were used: -c 750 Cy5

  -z/--zstacks
    The number of images collected in the z dimension.

Output:
The data organization table is printed out, and can be re-directed to a file

Examples:
For a MERFISH experiment with 16 bits, 3 colors, and 70 z-stacks:
  $python generate_data_organization.py -b 16 -c 750 Cy5 Cy3 -z 70 > dataorganization.csv
"""
import argparse

parser = argparse.ArgumentParser(description="Generate data organization files for MERlin")
parser.add_argument(
    "-b",
    "--bits",
    help="The number of bits in the MERFISH codebook",
    dest="bits",
    type=int,
    required=True,
)
parser.add_argument(
    "-c",
    "--colors",
    help="The imaging colors used (not including fiducial)",
    dest="colors",
    nargs="+",
    required=True,
)
parser.add_argument(
    "-z",
    "--zstacks",
    help="The number of z-dimension images/pixels",
    dest="zstacks",
    type=int,
    required=True,
)
args = parser.parse_args()

bits = args.bits
colors = args.colors
zstacks = args.zstacks

fiducial_color = "405"  # MERlin doesn't seem to use this for anything, so no need to change it
columns = "channelName,readoutName,imageType,imageRegExp,bitNumber,imagingRound,color,frame,zPos,fiducialImageType,fiducialRegExp,fiducialImagingRound,fiducialFrame,fiducialColor"
image_type = "Conv_zscan"
regexp = r"(?P<imageType>[\w|-]+)_H(?P<imagingRound>[0-9]+)_F_(?P<fov>[0-9]+)"
n_frames = zstacks * (len(colors) + 1)  # +1 for fiducial channel


def print_row(bit, name, hyb, color, frames, zpos):
    row = [
        name,
        name,
        image_type,
        regexp,
        bit,
        hyb,
        color,
        f'"{frames}"',
        f'"{zpos}"',
        image_type,
        regexp,
        hyb,
        len(colors),
        fiducial_color,
    ]
    print(",".join([str(x) for x in row]))


print(columns)
for bit in range(1, bits + 1):
    name = f"bit{bit}"
    hyb = str(((bit - 1) // len(colors)) + 1)
    color = colors[(bit - 1) % len(colors)]
    frames = str(list(range((bit - 1) % len(colors), n_frames, len(colors) + 1)))
    zpos = str(list(range(0, zstacks)))
    print_row(bit, name, hyb, color, frames, zpos)


name = "PolyT"
hyb = 0
color = 488
frames = str(list(range(3, n_frames, 5)))
zpos = str(list(range(0, zstacks)))
print_row("", name, hyb, color, frames, zpos)

name = "DAPI"
hyb = 0
color = 405
frames = str(list(range(4, n_frames, 5)))
zpos = str(list(range(0, zstacks)))
print_row("", name, hyb, color, frames, zpos)
