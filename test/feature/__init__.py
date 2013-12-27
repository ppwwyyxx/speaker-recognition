


#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: __init__.py
# Date: Fri Dec 27 02:52:42 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import BOB
import LPC
import numpy as np

def get_extractor(extract_func, **kwargs):
    def f(tup):
        return extract_func(*tup, **kwargs)
    return f

def mix_feature(tup):
    bob = BOB.extract(tup)
    lpc = LPC.extract(tup)
    return np.concatenate((bob, lpc), axis=1)
