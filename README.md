# speechlmsocre_tool

Implementation of "SpeechLMScore: Evaluating speech generation using speech langauge model" https://arxiv.org/abs/2212.04559


## Usage 

### Download pretrained models
* [Pretrained Hubert](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)  
* [Pretrained Hubert-kmeans](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin)
* [Speech ulm](https://huggingface.co/soumi-maiti/speech-ulm-lstm)


### Compute SpeechLMScore

Generates speechlmscore for each file in ```audio_dir``` in file ```ppl```

```
audio_dir=<folder containing audio>
layer=<Hubert layer to extract features>

./run.sh ${audio_dir} ${layer}
```
