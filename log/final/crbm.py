#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: crbm.py
# $Date: Sun Dec 29 22:48:01 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from dataextractor import DataExtractor as DE
import matplotlib.pyplot as plt

def populate_config(config):
    if 'ylabel' not in config:
        config['ylabel'] = 'Accuracy'
    return config

def main():
    configs = [
        dict(series=[
#            dict(path='crbm-log/30sec.real.testlog',
#                extractor=DE('nr_person=([0-9]+)', 'acc=([0-9\.]+)'),
#                label='30 sec training hahah'),
            dict(path='crbm-log/30sec.real.10test.testlog',
                extractor=DE('nr_person=([0-9]+)', 'acc=([0-9\.]+)'),
                label='30 sec training'),
            dict(path='crbm-log/30sec.all.log',
                extractor=DE('nr_person=([0-9]+)', 'acc=([0-9\.]+)'),
                label='60 sec training'),
            dict(path='crbm-log/nr_h-32.all.log',
                extractor=DE('nr_person=([0-9]+)', 'acc=([0-9\.]+)'),
                label='120 sec training'),
            ],
            title='Effect on number of speakers, using CRBM',
            xlabel='Number of Speakers',
            save_file='fig/crbm.pdf'),

    ]

    for cfg in configs:
        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        cfg = populate_config(cfg)
        y_min = 100
        for ser in cfg['series']:
            with open(ser['path']) as fin:
                lines = map(lambda x: x.rstrip(), fin.readlines())
            print ser['path']
            x, y, yerr = ser['extractor'].extract_data(lines)
            y_min = min(y_min, min(map(lambda p: p[0] - p[1], zip(y, yerr))))
#            plt.errorbar(x, y, yerr, lw=2, label=ser['label'])
            plt.plot(x, y, lw=2, label=ser['label'])
        plt.title(cfg['title'])
        plt.xlabel(cfg['xlabel'])
        plt.ylabel(cfg['ylabel'])
        if 'xlim' in cfg:
            plt.xlim(cfg['xlim'])
        plt.ylim([min(y_min, 0.9), 1.0])
        plt.legend(loc=3)
        ax.grid(color='gray', linestyle='dashed')
        plt.savefig(cfg['save_file'])


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
