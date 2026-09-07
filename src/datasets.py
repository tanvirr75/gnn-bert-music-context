import numpy as np
import torch
from torch.utils.data import Dataset


class TagDataset(Dataset):
    """
    Task 1: predict a clip's full tag set from a PARTIAL subset of its own
    tags. Showing all the true tags would let the model just copy the
    answer instead of learning anything.
    """

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