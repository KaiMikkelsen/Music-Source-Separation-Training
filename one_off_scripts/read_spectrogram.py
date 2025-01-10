import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import istft
from scipy.io.wavfile import write

# Load the spectrogram file
spectrogram_file_path = "/Users/kaimikkelsen/Downloads/spectrogram.npy"
spec = np.load(spectrogram_file_path)

# Define parameters
sampling_rate = 44100  # Match the original sampling rate
nperseg = 256  # Default window length used for spectrogram

# Perform the inverse STFT to reconstruct the audio
_, reconstructed_audio = istft(spec, fs=sampling_rate, nperseg=nperseg)

# Normalize the reconstructed audio to fit within -1 to 1 range
reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))

# Save the reconstructed audio file
reconstructed_audio_file_path = "/Users/kaimikkelsen/Downloads/reconstructed_audio.wav"
write(reconstructed_audio_file_path, sampling_rate, (reconstructed_audio * 32767).astype(np.int16))

# Plot the reconstructed audio waveform
plt.figure(figsize=(10, 4))
plt.plot(reconstructed_audio)
plt.title("Reconstructed Audio Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

print(f"Reconstructed audio saved to {reconstructed_audio_file_path}")