import numpy as np
from pylab import imshow, plot, figure, show
from scipy.signal import convolve
from TemplateSpeechRec.audio import make_gaussian_kernel

def test_smooth_spectrogram():
    """
    Tests what happens when we apply different signal
    processing things to images, to see how these things work
    """
    x = np.random.rand(2500).reshape(50,50)
    x[x>.90] = 1.
    x[x<=.90] = 0.
    imshow(x,interpolation="nearest")
    show()
    g = make_gaussian_kernel(7,3)
    smoothed_x = convolve(x,g,mode="same")
    # doesn't seem to be any border problems here

def test_spectrograms():
    x = np.sin(np.arange(3000)/3000.*2*np.pi*15)
    S = _spectrograms(x,320,5,512,3000,16000)
    S = _spectrograms(s,320,80,512,3000,16000)
