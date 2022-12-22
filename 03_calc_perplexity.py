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
from lm import LM
from dataloader import IterableDataset, collate_fn


from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("speech_ulm")


def calc_perplexity(
    prompts, token_list, config_file, ckpt_path, batch_size, out_dir, num_workers
):  

    # setup device 
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    
    # Load unit-lm
    model = LM.build_model_from_file(token_list, config_file, ckpt_path, device)

    # Load prompts
    # Build data-iterator
    dataset = IterableDataset(prompts, token_list)
    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size = batch_size, num_workers=num_workers)

    # 4. Start for-loop
    f_u = open(os.path.join(out_dir, "ppl"), "w", encoding="utf-8")
    ids = []
    lprobs = []
    total_ppl = 0.0
    total_utt = 0
    
    for keys, text, text_length in tqdm.tqdm(loader):
        _bs = len((text))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"

        with torch.no_grad():
            text = text.to(device)
            text_length = text_length.to(device)
            nll, lengths, hypo_lprobs = model.nll(text, text_length)
           
        for i, id in enumerate(keys):
            ids.append(id)
            lprobs.append(hypo_lprobs[i, :text_length[i], :].cpu().numpy().tolist())

        # nll: (B, L) -> (B,)
        nll = nll.detach().cpu().numpy().sum(1)
        # lengths: (B,)
        lengths = lengths.detach().cpu().numpy()

        for key, _nll, ntoken in zip(keys, nll, lengths):
            utt_ppl = (_nll / ntoken)
            
            f_u.write("{} {}\n".format(key, utt_ppl))

            total_ppl += utt_ppl 
            total_utt += 1
    
    f_u.close()

    # save total ppl
    print("Total ppl:", total_ppl/total_utt)
    dictionary = dict(zip(ids, lprobs))
    np.save(os.path.join(out_dir, "lprobs.npy"), dictionary)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="", required=True, help="")
    parser.add_argument("--token_list", type=str, default="", required=True, help="")
    parser.add_argument("--config_file", type=str, default="", required=True, help="")
    parser.add_argument("--ckpt_path", type=str, default="", required=True, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--out_dir", "-o", type=str, default="", required=True, help="")
    parser.add_argument("--num_workers", "-n", type=int, default=1, help="")
    

    args = parser.parse_args()

    logger.info(args)

    calc_perplexity(**vars(args))