import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

# Read the WAV file
sample_rate, samples = wavfile.read('/Users/kaimikkelsen/Downloads/pop_song.wav')

# If stereo, convert to mono by averaging the channels
if len(samples.shape) > 1:
    samples = samples.mean(axis=1)

# Calculate the spectrogram
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)


spectrogram_db = 10 * np.log10(spectrogram + 1e-10)  # add small value to avoid log(0)

# Plot with a perceptually uniform colormap
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, spectrogram_db, shading='gouraud', cmap='inferno')
plt.colorbar(label='Intensity [dB]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram (dB scale)')
plt.tight_layout()
plt.show()

#frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# plt.pcolormesh(times, frequencies, np.log(spectrogram))
# #plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

