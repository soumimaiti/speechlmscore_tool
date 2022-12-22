#!/bin/sh


audio_dir=$1 
l=$2
ext='.wav' #'.flac'
nj=1

echo "${audio_dir}"
split=$(basename ${audio_dir})"_"${l}
echo "${split}"

data_basedir="data"
hubert_path="facebook/hubert-base-ls960"
km_path="models/hubert/km_50.bin"
ulm_path="models/ulm/"
token_list="models/ulm/tokens.txt"

data_dir=${data_basedir}/${split}
mkdir -p ${data_dir}
mkdir -p ${data_dir}/log



# Get file list
python 01a_gen_list.py -a ${audio_dir} -o ${data_dir}/${split}_file_list --ext ${ext} > ./${data_dir}/log/log
# Gen tsv
(python 01b_gen_tsv.py -i ${data_dir}/${split}_file_list -o ${data_dir}/${split}.tsv) &>> ./${data_dir}/log/log
ret=$?
sync || true

# Get features - run on gpu
(python 02a_dump_feature.py --tsv_dir ${data_dir} \
    --split ${split} --ckpt_path ${hubert_path} --layer ${l} \
    --feat_dir ${data_dir} ) &> ./${data_dir}/log/dump_features.log
ret=$?
sync || true

# Get km labels - run on gpu
(python 02b_dump_km_label.py  --feat_dir ${data_dir} \
    --split ${split} --km_path ${km_path} \
    --lab_dir ${data_dir} --use_gpu  ) &> ./${data_dir}/log/dump_km_label.log
ret=$?
sync || true

# Creates keys file with file_id
sed '1d' ${data_dir}/${split}.tsv | awk '{n=split($1, lst, "/"); uttname=lst[n]; gsub(/\.wav|\.flac/, "", uttname); print(uttname)}' > ${data_dir}/${split}.keys
paste ${data_dir}/${split}.keys ${data_dir}/${split}.km > ${data_dir}/${split}.txt


batch_size=16
(python 03_calc_perplexity.py --prompts ${data_dir}/${split}.txt \
    --token_list ${token_list} \
    --config_file ${ulm_path}/config.yaml --ckpt_path ${ulm_path}/valid.loss.best.pth \
    --batch_size ${batch_size} --out_dir ${data_dir} -n ${nj} )&> ./${data_dir}/log/calc_ppl.log
ret=$?
sync || true
