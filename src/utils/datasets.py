import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader

from src.utils.augmentations import RotateAndReflect


class LCDataset(Dataset):

    def __init__(self, cfg, directory, device, use_labels=True, preprocessing=None):

        self.use_labels = use_labels
        self.files = sorted(glob(f"{directory}/run*.npz"))
        self.Xs, self.ys = [], []

        # dtype = torch.get_default_dtype()
        dtype = getattr(torch, cfg.dtype)

        for f in self.files:  # TODO: parallelize
            record = np.load(f)

            # read and preprocess
            X = torch.from_numpy(record["image"]).to(dtype)
            for f in preprocessing["x"]:
                X = f.forward(X)
            self.Xs.append(X)

            if use_labels:
                # read and preprocess
                y = torch.from_numpy(record["label"]).to(dtype)
                for f in preprocessing["y"]:
                    y = f.forward(y)
                self.ys.append(y)

            if cfg.on_gpu:
                X = X.to(device)
                if use_labels:
                    y = y.to(device)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return (self.Xs[idx], self.ys[idx]) if self.use_labels else (self.Xs[idx],)


class LCDatasetByFile(Dataset):

    def __init__(self, cfg, directory, use_labels=True, preprocessing=None):

        self.cfg = cfg
        self.use_labels = use_labels
        self.preprocessing = preprocessing

        self.files = sorted(glob(f"{directory}/run*.npz"))
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        record = np.load(self.files[idx])

        # read
        X = torch.from_numpy(record["image"]).to(self.dtype)
        # preprocess
        for f in self.preprocessing["x"]:
            X = f.forward(X)

        if self.use_labels:
            # read
            y = torch.from_numpy(record["label"]).to(self.dtype)
            # preprocess
            for f in self.preprocessing["y"]:
                y = f.forward(y)

        return (X, y) if self.use_labels else (X,)


class SummarizedLCDataset(Dataset):

    def __init__(
        self, dataset, summary_net, device, exp_cfg, dataset_cfg, augment=False
    ):

        self.Xs = []
        self.ys = []
        self.summary_net = summary_net

        self.pool_summary = not (
            hasattr(summary_net, "head")
            or exp_cfg.net.arch == "AttentiveHead"
            or (hasattr(exp_cfg, "use_attn_pool") and exp_cfg.use_attn_pool)
        )

        if augment:
            aug = RotateAndReflect(include_identity=True)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_cfg.summary_batch_size,
            num_workers=exp_cfg.num_cpus if exp_cfg.num_cpus > 1 else 0,
        )

        dset_device = device if dataset_cfg.on_gpu else torch.device("cpu")
        for X, y in dataloader:

            self.ys.append(y.to(dset_device))

            X = X.to(device)
            if augment:
                summary = torch.stack(  # collect all augmentations of X
                    [self.summarize(xa).to(dset_device) for xa in aug.enumerate(X)],
                    dim=1,
                )
            else:
                summary = self.summarize(X).to(dset_device)
            self.Xs.append(summary)

        self.Xs = torch.vstack(self.Xs)
        self.ys = torch.vstack(self.ys)

    @torch.no_grad()
    def summarize(self, x):
        x = self.summary_net(x)
        if self.pool_summary:
            x = x.mean(1)  # (B, T, D) --> (B, D)
        return x

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]

    def collate_fn(self, batch):
        """A collator that selects one augmentation of X."""
        # Same augmentation for each batch element. Alternative is to append in a loop
        X, y = torch.utils.data.default_collate(batch)
        idx = torch.randint(X.size(1), ())
        return X[:, idx], y
