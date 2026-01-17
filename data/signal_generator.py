import numpy as np

def qpsk_signal(n_symbols=1024):
    bits = np.random.randint(0, 2, size=(n_symbols, 2))
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    symbols /= np.sqrt(2)
    return symbols

def awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))
    )
    return signal + noise

def narrowband_interference(signal, freq=0.05, amplitude=0.7):
    t = np.arange(len(signal))
    interference = amplitude * np.exp(1j * 2 * np.pi * freq * t)
    return signal + interference

def impulsive_interference(signal, prob=0.01, amplitude=5):
    impulses = np.random.rand(len(signal)) < prob
    impulse_noise = impulses * amplitude * (
        np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))
    )
    return signal + impulse_noise

def generate_sample(label, n_symbols=1024):
    signal = qpsk_signal(n_symbols)

    if label == 0:
        return signal
    elif label == 1:
        return awgn(signal, snr_db=10)
    elif label == 2:
        return narrowband_interference(signal)
    elif label == 3:
        return impulsive_interference(signal)
    else:
        raise ValueError("Invalid label")

def generate_dataset(n_samples_per_class=500):
    X = []
    y = []

    for label in range(4):
        for _ in range(n_samples_per_class):
            s = generate_sample(label)
            X.append(np.vstack([s.real, s.imag]))
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__ == "__main__":
    X, y = generate_dataset()
    print("Dataset generated:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
