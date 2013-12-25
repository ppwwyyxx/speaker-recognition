#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Date: Wed Dec 25 16:17:07 2013 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


kwd_mark = object()

def cached_func(function):
    cache = {}
    def wrapper(*args, **kwargs):
        key = args + (kwd_mark,) + tuple(sorted(kwargs.items()))
        if key in cache:
            return cache[key]
        else:
            result = function(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper
