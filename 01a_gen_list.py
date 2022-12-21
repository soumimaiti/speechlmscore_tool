from os import listdir
from os.path import isfile, join, relpath
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', '-a', type=str, help='Path to input audio directory')
parser.add_argument('--out', '-o', type=str, help='Output file list path')
parser.add_argument('--ext', type=str, default=".wav", help='Audio file extenson')

args = parser.parse_args()


dataset_path = args.audio_dir
output_path = args.out


audio_files=[]
audio_files.extend(glob.glob(dataset_path+'/**/*{}'.format(args.ext), recursive=True))
audio_files.sort()

if len(audio_files) == 0:
    print("Error: No files found ")

print("Found files: ", len(audio_files))
with open(output_path, "w+") as f: 
   for idx, audio_file in enumerate(audio_files):
       f.write(f"{audio_file}\n")

print("Wrote file list to: ", output_path)
