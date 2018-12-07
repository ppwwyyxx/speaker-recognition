## About

This is a [Speaker Recognition](https://en.wikipedia.org/wiki/Speaker_recognition) system with GUI.

For more details of this project, please see:

+ Our [presentation slides](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Presentation.pdf)
+ Our [complete report](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Final-Report-Complete.pdf)

## Dependencies
+ Linux
+ [scikit-learn](http://scikit-learn.org/)
+ [scikits.talkbox](http://scikits.appspot.com/talkbox)
+ [pyssp](https://pypi.python.org/pypi/pyssp)
+ [PyQt4](http://sourceforge.net/projects/pyqt/)
+ [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/)
+ (Optional)Python bindings for [bob](http://idiap.github.io/bob/):
	+ install blitz, openblas, boost
	+ `pip install --user bob.extension bob.blitz bob.core bob.sp bob.ap`

Note: We have a MFCC implementation on our own
which will be used as a fallback when bob is unavailable.
But it's not so efficient as the C implementation in bob.

## Algorithms Used

_Voice Activity Detection_(VAD):
+ [Long-Term Spectral Divergence](http://www.sciencedirect.com/science/article/pii/S0167639303001201) (LTSD)

_Feature_:
+ [Mel-Frequency Cepstral Coefficient](http://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCC)
+ [Linear Predictive Coding](http://en.wikipedia.org/wiki/Linear_predictive_coding) (LPC)

_Model_:
+ [Gaussian Mixture Model](http://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) (GMM)
+ [Universal Background Model](http://www.sciencedirect.com/science/article/pii/S1051200499903615) (UBM)
+ Continuous [Restricted Boltzman Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (CRBM)
+ [Joint Factor Analysis](http://speech.fit.vutbr.cz/software/joint-factor-analysis-matlab-demo) (JFA)

## GUI Demo

Our GUI has basic functionality for recording, enrollment, training and testing, plus a visualization of real-time speaker recognition:

![graph](https://github.com/ppwwyyxx/speaker-recognition/raw/master/doc/Final-Report-Complete/img/gui-graph.png)

You can See our [demo video](https://github.com/ppwwyyxx/speaker-recognition/raw/master/demo.avi) (in Chinese).
Note that real-time speaker recognition is extremely hard, because we only use corpus of about 1 second length to identify the speaker.
Therefore the system doesn't work very perfect.

The GUI part is quite hacky for demo purpose and is not maintained anymore today. Take it as a reference, but don't expect it to work out of the box. Use command line tools to try the algorithms.

## Command Line Tools
```sh
usage: speaker-recognition.py [-h] -t TASK -i INPUT -m MODEL

Speaker Recognition Command Line Tool

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Task to do. Either "enroll" or "predict"
  -i INPUT, --input INPUT
                        Input Files(to predict) or Directories(to enroll)
  -m MODEL, --model MODEL
                        Model file to save(in enroll) or use(in predict)

Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob module.

Examples:
    Train:
    ./speaker-recognition.py -t enroll -i "./bob/ ./mary/ ./person*" -m model.out

    Predict:
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
```
