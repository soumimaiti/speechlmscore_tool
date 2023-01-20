## Training speech langauge model
Speech langauge model can be trained in two main steps: data preprocessing to get discrete tokens for training data and training. 

### Preprocessing Data
Given audio directory containing all training data, we can use preprocess data as following:

This step can be also performed using fairseq [example](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/ulm)


### Training speech language model
You can train a LSTM speech langauge model using ESPnet toolkit.

1. Create token list in ```data/tokens.txt'''.
```
  python create_token_list.txt
'''

3. Prepare data and move them to correct places
```
  mkdir -p data/token_list/word
  cp data/tokens.txt data/token_list/word/

  lm_train_text=data/train.txt
  data_feats=dump/raw
  mkdir -p ${data_feats}
  cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
  '''

3. You can use this run_lm.sh for training:
```
  #!/usr/bin/env bash
  # Set bash to 'debug' mode, it will exit on :
  # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
  set -e
  set -u
  set -o pipefail

  lm_train_text=data/train.txt
  lm_dev_text=data/valid.txt
  lm_test_text=data/test.txt

  lm_config=conf/train_lm_rnn_unit1024_nlayers3_dropout0.2_epoch30.yaml

  ./asr.sh \
      --stage 6 \
      --stop_stage 8 \
      --ngpu 1 \
      --nj 16 \
      --inference_nj 1 \
      --gpu_inference true \
      --train_set "dummy_train" \
      --valid_set "dummy_valid" \
      --test_sets "dummy_test" \
      --token_type word \
      --lm_config "${lm_config}" \
      --lm_train_text "${lm_train_text}" \
      --lm_dev_text "${lm_dev_text}" \
      --lm_test_text "${lm_test_text}" "$@"
'''
