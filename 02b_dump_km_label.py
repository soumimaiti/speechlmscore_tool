# The dumping k-means label code is inspired by code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_km_label.py
#


import argparse
import logging
import os
import sys

import joblib
import numpy as np
import torch
import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=str, default='0')
    parser.add_argument("--lab_dir", type=str, required=True)
    parser.add_argument("--use_gpu", action='store_true')
    
    return parser


class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank):

    if nshard > 1:
        feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
        leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    else:
        feat_path = f"{feat_dir}/{split}.npy"
        leng_path = f"{feat_dir}/{split}.len"

    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset : offset + leng]

    return iterate, len(lengs)


def dump_label(feat_dir, split, km_path, nshard, rank, lab_dir, use_gpu):
    apply_kmeans = ApplyKmeans(km_path, use_gpu=use_gpu)
    generator, num = get_feat_iterator(feat_dir, split, nshard, rank)
    iterator = generator()

    if nshard > 1:
        lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
    else:
        lab_path = f"{lab_dir}/{split}.km"

    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.rank = eval(args.rank)
    logging.info(str(args))

    dump_label(**vars(args))
