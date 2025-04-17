import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

# Read the WAV file
sample_rate, samples = wavfile.read('/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/one_off_scripts/piano_chords.wav')

# If stereo, convert to mono by averaging the channels
if len(samples.shape) > 1:
    samples = samples.mean(axis=1)

# Calculate the spectrogram
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, np.log(spectrogram))
#plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

