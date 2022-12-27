# speechlmscore_tool

Implementation of "SpeechLMScore: Evaluating speech generation using speech langauge model" https://arxiv.org/abs/2212.04559


## Installation
You should be able to install using
```
python setup.py install
```

## Usage 

### Download pretrained models
Download these pretrained models and update their path in ```run.sh```. Note: tokens.txt is located with speech ulm model.

* [Pretrained Hubert](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)  
* [Pretrained Hubert-kmeans](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin)
* [Speech ulm](https://huggingface.co/soumi-maiti/speech-ulm-lstm)


### Compute SpeechLMScore

Generates speechlmscore for each file in ```audio_dir``` in file ```ppl```. Note: for using audio files other than ```.wav``` set ext variable is run.sh. Audio files with sampling rate of 16kHZ are supported.

```
audio_dir=<folder containing audio>
layer=<Hubert layer to extract features>

./run.sh ${audio_dir} ${layer}
```
