import sys
import pathlib
import re
from collections import defaultdict

folder = pathlib.Path(sys.argv[1])
print(f"Verifying {folder}...")

sizes = defaultdict(lambda: defaultdict(int))

count = 0
rounds = set()
fovs = set()
for dax in folder.glob("*.dax"):
    size = dax.stat().st_size
    match = re.search("_H([0-9]+)_F_?([0-9]+).dax", str(dax))
    round = int(match.group(1))
    fov = int(match.group(2))
    sizes[round][size] += 1
    count += 1
    rounds.add(round)
    fovs.add(fov)

print(f"INFO: Found {count} images across {len(rounds)} imaging rounds and {len(fovs)} FOVs")

nums = set()
for round in sizes.values():
    nums.add(len(round))

if len(nums) == 1:
    print("PASS: Every image in each round is the same size")
else:
    print("FAIL: Some images are different sizes, possible incomplete copy")

nums = set()
for round in sizes.values():
    nums.add(list(round.values())[0])

if len(nums) == 1:
    print(f"PASS: Every imaging round has {nums.pop()} FOVs")
else:
    print("FAIL: Not all imaging rounds have the same number of FOVs, possible missing files")
