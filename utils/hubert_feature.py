# The hubert feature reader code is inspired by code from Fairseq:
#     https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py
#

import logging
import os
import sys
from transformers import HubertModel
import soundfile as sf
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")

class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
            
        model = HubertModel.from_pretrained(ckpt_path)
        self.model = model.eval().cuda()
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")
        
    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav
    
    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda().unsqueeze(0)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                
                feat_chunks = self.model(x_chunk,output_hidden_states=True).hidden_states
                feat_chunk = feat_chunks[self.layer]

                feat.append(feat_chunk)
            return torch.cat(feat, 1).squeeze(0)
