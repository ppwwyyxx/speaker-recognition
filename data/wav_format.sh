#!/bin/bash -e
# File: wav_format.sh
# Date: Sun Nov 17 14:44:02 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

[[ -z "$1" || -z "$2" ]] && (echo "Usage: $0 <input> <output>" && exit 1)

mplayer $1 -ao pcm:file=$2 -loop 1
