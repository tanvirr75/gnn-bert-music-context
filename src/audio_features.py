import librosa


def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def get_melspectrogram(y, sr, n_mels=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=1.0).T
    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True) + 1e-8
    return (log_mel - mean) / std


def get_chroma(y, sr, n_chroma=12):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma).T
    mean = chroma.mean(axis=0, keepdims=True)
    std = chroma.std(axis=0, keepdims=True) + 1e-8
    return (chroma - mean) / std


def split_into_segments(feature_matrix, sr, hop_length=512, window_seconds=5.0):
    frames_per_segment = max(1, int(window_seconds * sr / hop_length))
    segments = []
    for start in range(0, feature_matrix.shape[0], frames_per_segment):
        chunk = feature_matrix[start:start + frames_per_segment]
        if len(chunk) > 0:
            segments.append(chunk.mean(axis=0))
    return segments