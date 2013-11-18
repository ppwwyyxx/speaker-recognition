#!/bin/bash -e
# File: convert_all.sh
# Date: Sun Nov 17 17:05:12 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

mkdir -p $1/converted
for i in $1/*.wav; do
	echo $i
	$(dirname $BASH_SOURCE)/wav_format.sh $i $1/converted/$i 2>/dev/null 1>/dev/null
done
