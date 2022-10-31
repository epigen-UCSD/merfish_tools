#!/bin/bash
# Usage: archive_directory.sh folder1 [folder2 folder3 ...]
# This script will create a .tar.gz file for each folder given using pigz for
# parallel compression. The original folder will be deleted once the archive
# is created.

for dir in $@
do
	(tar -cvf - $dir/../$(basename $dir) | pigz -9 -p 32 > ${dir%/}.tar.gz) && rm -rf $dir
done
