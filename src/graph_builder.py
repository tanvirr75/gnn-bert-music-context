import numpy as np


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