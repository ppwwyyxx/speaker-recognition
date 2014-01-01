#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: plot-gmm.py
# $Date: Tue Dec 10 16:14:53 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
from scipy import stats, mgrid, c_, reshape, random, rot90
import argparse
from numpy import *
import numpy as np

class GassianTypeNotImplemented(Exception):
    pass

def get_args():
    description = 'plot gmm'
    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('-i', '--input', help = 'data file', required = True)
    parser.add_argument('-m', '--model', help = 'model file', required = True)

    args = parser.parse_args()

    return args


class Gaussian(object):
    def __init__(self):
        self.covtype = 1
        self.dim = 0
        self.mean = array([])
        self.sigma = array([])
        self.covariance = array([[]])

    def probability_of(self, x):
        assert len(x) == self.dim

        return exp((x - mean)**2 / (2 * self.sigma**2)) / (2 * pi * self.sigma)


class GMM(object):
    def __init__(self):
        self.nr_mixtures = 0
        self.weights = array([])
        self.gaussians = []

def read_data(fname):
    with open(fname) as fin:
        return zip(*map( lambda line: map(float, line.rstrip().split()), fin))


def read_gaussian(fin):
    gaussian = Gaussian()
    gaussian.dim, gaussian.covtype = map(int, fin.readline().rstrip().split())
    if gaussian.covtype == 1:
        gaussian.mean = map(float, fin.readline().rstrip().split())
        gaussian.sigma = map(float, fin.readline().rstrip().split())
        assert len(gaussian.mean) == gaussian.dim
        assert len(gaussian.sigma) == gaussian.dim
    else:
        raise GassianTypeNotImplemented()
    return gaussian

def read_model(fname):
    gmm = GMM()
    with open(fname) as fin:
        gmm.nr_mixtures = int(fin.readline().rstrip())
        gmm.weights = map(float, fin.readline().rstrip().split())
        for i in range(gmm.nr_mixtures):
            gmm.gaussians.append(read_gaussian(fin))

    return gmm

def main():
    args = get_args()
    data = read_data(args.input)
    x, y = data[:2]
    gmm =  read_model(args.model)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 'equal')
    ax.scatter(x, y)
    x0, x1, y0, y1 = ax.axis()

    x = linspace(x0, x1, 1000)
    y = linspace(y0, y1, 1000)
    X, Y = meshgrid(x, y)

    def get_Z(X, Y, gaussian):
        return mlab.bivariate_normal(X, Y, gaussian.sigma[0], gaussian.sigma[1],
                gaussian.mean[0], gaussian.mean[1], 0)

    Z = get_Z(X, Y, gmm.gaussians[0])
    for gaussian in gmm.gaussians[1:]:
        Z += get_Z(X, Y, gaussian)
    plt.contour(X, Y, Z, cmap=cm.PuBu_r)
    for gaussian in gmm.gaussians:
#        print gaussian.mean
        plt.scatter(gaussian.mean[0], gaussian.mean[1], s = 50, c = 'yellow')

    plt.show()


if __name__ == '__main__':
    main()


# vim: foldmethod=marker

