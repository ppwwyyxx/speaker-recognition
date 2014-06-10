#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: speaker-recognition.py
# Date: Tue Jun 10 17:08:13 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import sys
import glob
import os
import itertools
import scipy.io.wavfile as wavfile

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'gui'))
from gui.interface import ModelInterface
from filters.silence import remove_silence

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Note that wildcard inputs should be *quoted*, and they will be sent to glob

Examples:
    Train:
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out

    Predict:
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll", "predict"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=True)

    ret = parser.parse_args()
    return ret

def task_enroll(input_dirs, output_model):
    m = ModelInterface()
    input_dirs = input_dirs.strip().split()
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = []
    for d in dirs:
        label = os.path.basename(d)

        wavs = glob.glob(d + '/*.wav')
        if len(wavs) == 0:
            continue
        print "Label {0} has files {1}".format(label, ','.join(wavs))
        for wav in wavs:
            fs, signal = wavfile.read(wav)
            m.enroll(label, fs, signal)

    m.train()
    m.dump(output_model)

def task_predict(input_files, input_model):
    m = ModelInterface.load(input_model)
    for f in glob.glob(input_files):
        fs, signal = wavfile.read(f)
        label = m.predict(fs, signal)
        print f, '->', label

if __name__ == '__main__':
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
