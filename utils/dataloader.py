from typing import Tuple, Union
import logging, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from torch.utils.data import IterableDataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dataloader")


# Token to id converter
class TokenIDConverter:
    def __init__(
        self,
        tokens_filepath,
        unk_symbol="<unk>",
    ):
        self.token_list = []
        with open(tokens_filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip()
                self.token_list.append(line)

        self.token2id = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self):
        return len(self.token_list)

    def tokens2ids(self, tokens):
        return [self.token2id.get(i, self.unk_id) for i in tokens]


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def collate_fn(
    data,
    pad_value = 0
    ):
    

    keys = [d[0] for d in data]
    tokens = [d[1] for d in data]
    token_lengths =  [d[1].shape[0] for d in data]

    tensor_list = [torch.from_numpy(a) for a in tokens]
    tokens = pad_list(tensor_list, pad_value)
    token_lengths = torch.tensor(token_lengths, dtype=torch.int)

    return (keys, tokens, token_lengths)


class IterableDataset(IterableDataset):

    def __init__(self, filename_utt, tokens_filepath):
        self.filename_utt = filename_utt
        self.token_id_converter = TokenIDConverter(tokens_filepath)

    def line_mapper(self, line):
        line_split = line.split('\t')
        
        key = line_split[0]
        tokens = line_split[1]
        key = key.strip()
        
        # token type -- word
        tokens=tokens.strip().split()
        token_ids = self.token_id_converter.tokens2ids(tokens)
        token_ids = np.array(token_ids, dtype=np.int64)
        
        return key, token_ids


    def __iter__(self):
        # Create an iterator
        itr = open(self.filename_utt)
        
        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, itr)
        
        return mapped_itr



