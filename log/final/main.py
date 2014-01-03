#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Fri Jan 03 22:01:46 2014 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

from dataextractor import DataExtractor as DE
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def populate_config(config):
    if 'ylabel' not in config:
        config['ylabel'] = 'Accuracy'
    return config

def main():
    configs = [
        dict(series=[
            dict(path='final-log/bob-nceps.log',
                extractor=DE('^n_ceps=([0-9]+)'),
                label=''),
            ],
            xlabel='Number of Cepstrals',
            save_file='fig/mfcc-nceps.pdf'),
        dict(series=[
            dict(path='final-log/bob-nfilter.log',
                extractor=DE('^MFCC NFILTER=([0-9]+)'),
                label=''),
            ],
            xlabel='Number of Filters',
            save_file='fig/mfcc-nfilter.pdf'),
        dict(series=[
            dict(path='final-log/bob-win.log',
                extractor=DE('WINL=([0-9]+)'),
                label=''),
            ],
            xlabel='Frame Length',
            save_file='fig/mfcc-frame-len.pdf'),

        dict(series=[
            dict(path='final-log/lpc-dim.log',
                extractor=DE('n_dim=([0-9]+)'),
                label=''),
            ],
            xlabel='Number of Coefficients',
            save_file='fig/lpc-nceps.pdf'),

        dict(series=[
            dict(path='final-log/lpc-win.log',
                extractor=DE('WINL=([0-9]+)'),
                label=''),
            ],
            xlabel='Frame Length',
            save_file='fig/lpc-frame-len.pdf'),

        dict(series=[
            dict(path='final-log/nmixture.log',
                extractor=DE('^nmixture:[^0-9]*([0-9]+)'),
                label=''),
            ],
            xlabel='Number of Mixtures',
            save_file='fig/nmixture.pdf'),

        dict(series=[
            dict(path='final-log/nmixture.log',
                extractor=DE('^nmixture:[^0-9]*([0-9]+)'),
                label=''),
            ],
            xlabel='Number of Mixtures',
            save_file='fig/nmixture.pdf'),

        #dict(series=[
            #dict(path='final-log/nperson-sklearn-mix-t3.log',
                #extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                #label='3s'),
            #dict(path='final-log/nperson-sklearn-mix-t4.log',
                #extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                #label='4s'),
            #dict(path='final-log/nperson-sklearn-mix-t5.log',
                #extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                #label='5s'),
            #], title='Effect on number of speakers and duration of test utterance',
            #xlabel='Number of Speakers',
            #save_file='fig/performance.pdf'),

        dict(series=[
            dict(path='final-log/nperson-mix-1523-t2.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='2s'),
            dict(path='final-log/nperson-mix-1523-t3.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='3s'),
            dict(path='final-log/nperson-mix-1523-t4.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='4s'),
            dict(path='final-log/nperson-mix-1523-t5.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='5s'),
            ], title='Effect on number of speakers and duration ' + \
                    'of test utterance, Reading style',
            xlabel='Number of Speakers (Reading)',
            save_file='fig/reading.pdf'),

        dict(series=[
            dict(path='final-log/nperson-mix-1523-t2-spont.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='2s'),
            dict(path='final-log/nperson-mix-1523-t3-spont.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='3s'),
            dict(path='final-log/nperson-mix-1523-t4-spont.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='4s'),
            dict(path='final-log/nperson-mix-1523-t5-spont.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='5s'),
            ], title='Effect on number of speakers and duration ' + \
                    'of test utterance, Spontaneous style',
            xlabel='Number of Speakers (Spontaneous)',
            save_file='fig/spont.pdf'),


        dict(series=[
            dict(path='final-log/nperson-mix-1523-t2-whisper.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='2s'),
            dict(path='final-log/nperson-mix-1523-t3-whisper.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='3s'),
            dict(path='final-log/nperson-mix-1523-t4-whisper.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='4s'),
            dict(path='final-log/nperson-mix-1523-t5-whisper.log',
                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
                label='5s'),
            ], title='Effect on number of speakers and duration ' + \
                    'of test utterance, Whisper style',
            xlabel='Number of Speakers (Whisper)',
            save_file='fig/whisper.pdf'),

#        dict(series=[
#            dict(path='final-log/nperson-newg-mix-t4.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='4s'),
#            dict(path='final-log/nperson-newg-mix-t5.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='5s'),
#            ], title='Effect on number of speakers and duration of test utterance, using our GMM',
#            xlabel='Number of Speakers',
#            save_file='fig/newgmm.pdf'),


#        dict(series=[
#            dict(path='final-log/nperson-newg-mix-t4.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='4s our GMM'),
#            dict(path='final-log/nperson-newg-mix-t5.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='5s our GMM'),
#            dict(path='final-log/nperson-sklearn-mix-t3.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='3s sklearn'),
#            dict(path='final-log/nperson-sklearn-mix-t4.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='4s sklearn'),
#            dict(path='final-log/nperson-sklearn-mix-t5.log',
#                extractor=DE('^Nperson:[^0-9]*([0-9]+)'),
#                label='5s sklearn'),
#            ], title='Effect on number of speakers and duration of test utterance',
#            xlabel='Number of Speakers',
#            save_file='fig/gmm-comparason.pdf'),
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
        plt.xlabel(cfg['xlabel'], fontsize=13)
        plt.ylabel(cfg['ylabel'], fontsize=13)
        plt.yticks(fontsize=15)
        if 'xlim' in cfg:
            plt.xlim(cfg['xlim'])
        plt.ylim([min(y_min, 0.9), 1.0])
        plt.legend(loc=3)
        ax.grid(color='gray', linestyle='dashed')
        plt.savefig(cfg['save_file'])


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
