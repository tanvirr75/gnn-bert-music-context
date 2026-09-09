import numpy as np
import torch
from torch_geometric.data import Data


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def build_segment_graph(segments, similarity_threshold=0.5):
    n = len(segments)
    edges = set()
    for i in range(n - 1):
        edges.add((i, i + 1))
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(segments[i], segments[j]) > similarity_threshold:
                edges.add((i, j))
    return list(edges)


def process_and_cache_clip(mp3_path, save_path, audio_dir="", window_seconds=5.0):

    from audio_features import load_audio, get_melspectrogram, split_into_segments

    full_path = f"{audio_dir}/{mp3_path}" if audio_dir else mp3_path
    y, sr = load_audio(full_path)
    mel = get_melspectrogram(y, sr)
    segments = split_into_segments(mel, sr, window_seconds=window_seconds)
    edges = build_segment_graph(segments)

    x = torch.tensor(np.array(segments), dtype=torch.float32)
    if len(edges) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    torch.save(Data(x=x, edge_index=edge_index), save_path)