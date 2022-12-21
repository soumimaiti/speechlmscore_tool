from os import listdir
from os.path import isfile, join
import argparse
import glob
import sox
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file_list', '-i', type=str, help='Path to input wav directory')
parser.add_argument('--out', '-o', type=str, help='Output tsv file path')
args = parser.parse_args()

output_path = args.out
    
audio_files=[]
with open(args.input_file_list) as f:
    for line in f:
        audio_files.append(line.strip())
audio_files.sort()


with open(output_path, "w") as f:
    f.write("."+"\n") # if relative path is used
    #f.write(" "+"\n") # if absolute path is used
    for audio_file in tqdm.tqdm(audio_files):
        n_sample = sox.file_info.num_samples(audio_file)
        f.write(f"{audio_file}\t{str(n_sample)}\n")