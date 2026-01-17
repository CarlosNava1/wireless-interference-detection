import numpy as np
import matplotlib.pyplot as plt

from data.signal_generator import generate_sample

def plot_signal_and_spectrum(label, title):
    signal = generate_sample(label)

    t = np.arange(len(signal))

    plt.figure(figsize=(12, 6))

    # Time domain (I)
    plt.subplot(2, 1, 1)
    plt.plot(t, signal.real)
    plt.title(f"{title} - Time Domain (I)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # Frequency domain
    plt.subplot(2, 1, 2)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freq = np.linspace(-0.5, 0.5, len(spectrum))
    plt.plot(freq, np.abs(spectrum))
    plt.title(f"{title} - Frequency Domain")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_signal_and_spectrum(0, "Clean QPSK")
    plot_signal_and_spectrum(1, "QPSK with AWGN")
    plot_signal_and_spectrum(2, "QPSK with Narrowband Interference")
    plot_signal_and_spectrum(3, "QPSK with Impulsive Interference")
