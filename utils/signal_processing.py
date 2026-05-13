import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# -----------------------------
# Signal utilities
# -----------------------------
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def create_windows(signal, window_size=2048):
    return np.array([
        signal[i:i+window_size]
        for i in range(0, len(signal)-window_size, window_size)
    ])


def bandpass_filter(signal, fs=48000, low=500, high=10000):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)


def compute_fft_features(windows):
    filtered = np.array([bandpass_filter(w) for w in windows])
    envelope = np.abs(hilbert(filtered, axis=1))
    fft =np.abs(np.fft.rfft(envelope, axis=1)) 
    return np.log1p(fft)

def ema(signal, alpha=0.1):
    smoothed = [signal[0]]
    for i in range(1, len(signal)):
        smoothed.append(alpha * signal[i] + (1 - alpha) * smoothed[i - 1])
    return np.array(smoothed) 