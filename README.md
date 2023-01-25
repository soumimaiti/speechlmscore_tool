# speechlmscore_tool

Implementation of "SpeechLMScore: Evaluating speech generation using speech langauge model" https://arxiv.org/abs/2212.04559


## Installation
You can install required python packages as:
```
python setup.py install
```

## Usage 

### Download pretrained models
Download these pretrained models and update their path in ```run.sh```.  
Note: ```tokens.txt``` is located with speech ulm model.

* [Pretrained Hubert](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)  
* [Pretrained Hubert-kmeans](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin)
* [Speech ulm](https://huggingface.co/soumi-maiti/speech-ulm-lstm)

Run the following command to download all the above models:
```shell
./download_pretrained_models.sh
```

### Compute SpeechLMScore using pretrained models

Generates speechlmscore for each file in ```audio_dir``` in file ```ppl```.  
Audio files with sampling rate of ```16kHZ``` are supported.  
Note: for using audio files other than ```.wav``` set ext variable is run.sh.  

```
audio_dir=<folder containing audio>
layer=<Hubert layer to extract features>

./run.sh ${audio_dir} ${layer}
```


### Train speech language models
Additionally speech language model can be trained and used for evaluation as well. [More Details](https://github.com/soumimaiti/speechlmscore_tool/Training.md)
