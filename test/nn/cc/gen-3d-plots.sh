#!/bin/bash
output_dir=3d-plots
lines=4000
fields="1,2,3"
xlim="-3,3"
ylim="-3,3"
zlim="-3,3"
for style in Style_Reading Style_Spontaneous Style_Whisper
do
	for mfcc_file in mfcc-data/$style/*.mfcc
	do
		echo $mfcc_file
		bname=`basename $mfcc_file`
		fbase=${bname%.*}
		output_fname=$output_dir/$style/$fbase.png
		mkdir -p $(dirname $output_fname)
		head $mfcc_file -n $lines \
			| cut -d ' ' -f $fields \
			| ./plot-point-3d.py -i '$stdin$' --output $output_fname \
			--xlim=$xlim --ylim=$ylim --zlim=$zlim
	done
done
