#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: dataextractor.py
# $Date: Sun Dec 29 21:45:53 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import re
from collections import defaultdict
import numpy as np

class DataExtractor(object):

    x_regex = None
    y_regex = None

    def __init__(self, x_regstr, y_regstr = '^[0-9]+\/[0-9]+ (.*)$'):
        self.x_regex = re.compile(x_regstr)
        self.y_regex = re.compile(y_regstr)

    def get_x(self, line):
        """get an x coordinate, could duplicate during run"""
        match = self.x_regex.findall(line)
        if len(match) == 1:
            return float(match[0])
        return None

    def get_y(self, line):
        """get_x and get_y should activate in turn, with x first"""
        match = self.y_regex.findall(line)
        if len(match) == 1:
            return float(match[0])
        return None

    def extract_data(self, lines):
        """ return x, y, yerr """
        data = defaultdict(list)
        cur_x = None
        for lino, line in enumerate(lines):
            tx = self.get_x(line)
            if tx is not None:
                assert cur_x is None, (lino + 1, line)
                cur_x = tx

            ty = self.get_y(line)
            if ty is not None:
                assert cur_x is not None, (lino + 1, line)
                data[cur_x].append(ty)
                cur_x = None

        xs, ys, yerr = [], [], []
        for x, y in sorted(data.iteritems()):
            xs.append(x)
            ys.append(np.mean(y))
            yerr.append(np.std(y))

        return xs, ys, yerr

# vim: foldmethod=marker
