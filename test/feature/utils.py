#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Date: Wed Dec 25 15:21:07 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


def cached_func(function):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            result = function(*args)
            cache[args] = result
            return result
    return wrapper
