import numpy as np
import torch
from torch.utils.data import Dataset


class TagDataset(Dataset):
    """Task 1: predict full tag set from a PARTIAL subset of a clip's own tags."""
    def __init__(self, df, tag_columns, tokenizer, max_length=32, mask_ratio=0.5, seed=42):
        self.df = df.reset_index(drop=True)
        self.tag_columns = tag_columns
        self.tokenizer = tokenizer
        self.max_length = max_length

        rng = np.random.RandomState(seed)
        self.captions = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            true_tags = [tag for tag in tag_columns if row[tag] == 1]
            if not true_tags:
                self.captions.append("[no tags]")
                continue
            n_keep = max(1, round(len(true_tags) * (1 - mask_ratio)))
            kept = rng.choice(true_tags, size=n_keep, replace=False)
            self.captions.append(" ".join(kept))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = self.tokenizer(
            self.captions[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        label = row[self.tag_columns].values.astype("float32")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
        }


class GraphTagDataset(torch.utils.data.Dataset):
    """Task 2: cached MTAT graph + multi-hot tag label."""
    def __init__(self, df, tag_columns):
        self.df = df.reset_index(drop=True)
        self.tag_columns = tag_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/{row['clip_id']}.pt", weights_only=False)
        graph.y = torch.tensor(row[self.tag_columns].values.astype("float32")).unsqueeze(0)
        return graph


class SegmentMatrixDataset(torch.utils.data.Dataset):
    """Task 2, B2: fixed-size segment matrix (crop/pad) for the CNN baseline."""
    def __init__(self, df, tag_columns, max_segments=8):
        self.df = df.reset_index(drop=True)
        self.tag_columns = tag_columns
        self.max_segments = max_segments

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/{row['clip_id']}.pt", weights_only=False)
        x = graph.x.numpy()
        n = x.shape[0]
        if n >= self.max_segments:
            x = x[:self.max_segments]
        else:
            x = np.vstack([x, np.zeros((self.max_segments - n, x.shape[1]), dtype=np.float32)])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row[self.tag_columns].values.astype("float32"))
        return x, label


class GraphGenreDataset(torch.utils.data.Dataset):
    """GTZAN: cached graph + single genre label, for the GNN."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/gtzan/{row['clip_id']}.pt", weights_only=False)
        graph.y = torch.tensor([row["label"]], dtype=torch.long)
        return graph


class GTZANSegmentDataset(torch.utils.data.Dataset):
    """GTZAN: fixed-size segment matrix, for the CNN baseline."""
    def __init__(self, df, max_segments=8):
        self.df = df.reset_index(drop=True)
        self.max_segments = max_segments

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/gtzan/{row['clip_id']}.pt", weights_only=False)
        x = graph.x.numpy()
        n = x.shape[0]
        if n >= self.max_segments:
            x = x[:self.max_segments]
        else:
            x = np.vstack([x, np.zeros((self.max_segments - n, x.shape[1]), dtype=np.float32)])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.long)
        return x, label


class FusionTagDatasetV2(torch.utils.data.Dataset):
    """Task 3: MTAT graph + partial-tag pseudo-caption + tag label. has_emotion=False."""
    def __init__(self, df, tag_columns, tokenizer, max_length=32, mask_ratio=0.5, seed=42):
        self.df = df.reset_index(drop=True)
        self.tag_columns = tag_columns
        self.tokenizer = tokenizer
        self.max_length = max_length

        rng = np.random.RandomState(seed)
        self.captions = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            true_tags = [tag for tag in tag_columns if row[tag] == 1]
            if not true_tags:
                self.captions.append("[no tags]")
                continue
            n_keep = max(1, round(len(true_tags) * (1 - mask_ratio)))
            kept = rng.choice(true_tags, size=n_keep, replace=False)
            self.captions.append(" ".join(kept))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/{row['clip_id']}.pt", weights_only=False)
        tokens = self.tokenizer(
            self.captions[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        graph.input_ids = tokens["input_ids"]
        graph.attention_mask = tokens["attention_mask"]
        graph.y = torch.tensor(row[self.tag_columns].values.astype("float32")).unsqueeze(0)
        graph.emotion = torch.zeros(1, 2)
        graph.has_emotion = torch.tensor([False])
        return graph


class FusionEmotionDataset(torch.utils.data.Dataset):
    """Task 3: DEAM graph + placeholder caption + valence/arousal. has_emotion=True."""
    def __init__(self, df, tokenizer, max_length=32):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/deam/{row['song_id']}.pt", weights_only=False)
        tokens = self.tokenizer(
            "[no tags]", padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        graph.input_ids = tokens["input_ids"]
        graph.attention_mask = tokens["attention_mask"]
        graph.y = torch.zeros(1, 50)
        graph.emotion = torch.tensor([[row["valence_mean"], row["arousal_mean"]]], dtype=torch.float32)
        graph.has_emotion = torch.tensor([True])
        return graph


class MusicCapsDataset(torch.utils.data.Dataset):
    """Task 4: MusicCaps graph + real natural-language caption."""
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = torch.load(f"data/processed/musiccaps/{row['clip_id']}.pt", weights_only=False)
        tokens = self.tokenizer(
            row["caption"], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        graph.input_ids = tokens["input_ids"]
        graph.attention_mask = tokens["attention_mask"]
        return graph