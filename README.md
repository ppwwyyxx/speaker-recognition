## Introduction

This is a speaker-recognition system with GUI, served as an SRT project for the course *Signal Processing* (2013fall) in Tsinghua Univ.

## Dependencies

+ [SciPy](http://www.scipy.org/)
+ [scikit-learn](http://scikit-learn.org/)
+ [scikits.talkbox](http://scikits.appspot.com/talkbox)
+ [bob](http://idiap.github.io/bob/)
+ [pyssp](https://pypi.python.org/pypi/pyssp)
+ [PyQt](http://sourceforge.net/projects/pyqt/)
+ [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/)
+ gcc >= 4.7

## Algorithm Used

_Voice Activity Detection_(VAD): Energy-Based, Long-Term Spectral Divergence(LTSD)

_Feature_: Mel-Frequency Cepstral Coefficient(MFCC), Linear Prediction Coding(LPC)

_Model_: Gaussian Mixture Model(GMM), Universal Background Model(UBM), Continuous Restricted Boltzman Machine(CRBM), Joint Factor Analysis(JFA)

For more details of this project, please see:

+ Our [presentation slides](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Presentation.pdf)
+ Our [complete report](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Final-Report-Complete.pdf)

## GUI demo

Our GUI not only has basic functionality for recording, enrollment, training and testing, but also has a visualization of real-time speaker recognition:

![graph](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Final-Report-Complete/img/gui-graph.png)

See our [demo video](https://github.com/ppwwyyxx/speaker-recognition/raw/master/demo.avi) (in Chinese) for more details.
