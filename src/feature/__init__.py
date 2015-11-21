


#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: __init__.py
# Date: Sat Nov 29 21:42:15 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
try:
    import BOB as MFCC
except Exception as e:
    print >> sys.stderr, e
    print >> sys.stderr, "Warning: failed to import Bob, will use a slower version of MFCC instead."
    import MFCC
import LPC
import numpy as np

def get_extractor(extract_func, **kwargs):
    def f(tup):
        return extract_func(*tup, **kwargs)
    return f

def mix_feature(tup):
    mfcc = MFCC.extract(tup)
    lpc = LPC.extract(tup)
    if len(mfcc) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(tup[1])
    return np.concatenate((mfcc, lpc), axis=1)
