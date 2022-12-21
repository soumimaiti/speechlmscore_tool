# The dump_km_label.py is inspired by code from Fairseq:
#     https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py
#


import numpy as np
import torch
import logging
import os
import sys
sys.path.append('utils')

import argparse
import math
from npy_append_array import NpyAppendArray
import soundfile as sf
import tqdm
from hubert_feature import HubertFeatureReader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


def get_path_iterator(tsv, nshard, rank):
    
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, "start={start}, end={end}"
        logger.info(
            f"rank {rank} of {nshard}, process {end-start} "
            f"({start}-{end}) out of {tot}"
        )

        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

        return iterate, len(lines)

def dump_feature_hubert(
    tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk
):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    logger.info("path iterator: nshard {} rank: {}".format(nshard, rank))
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    iterator = generator()

    if nshard > 1:
        feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
        leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    else:
        feat_path = f"{feat_dir}/{split}.npy"
        leng_path = f"{feat_dir}/{split}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_dir", type=str, default="", required=True, help="")
    parser.add_argument("--split", type=str, default="", required=True, help="")
    parser.add_argument("--ckpt_path", type=str, default="", required=True, help="")
    parser.add_argument("--layer", type=int, default=None, required=True, help="")
    parser.add_argument("--nshard", type=int, default=1, help="")
    parser.add_argument("--rank", type=str, default='0', help="")
    parser.add_argument("--feat_dir", type=str, default="", required=True, help="")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    
    args = parser.parse_args()
    
    args.rank = eval(args.rank)

    logger.info(args)

    dump_feature_hubert(**vars(args))