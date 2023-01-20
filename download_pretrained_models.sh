
# Preparing hubert_base_ls960
mkdir -p facebook
cd facebook
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
cd ..

# preparing km
mkdir -p models/hubert
cd models/hubert
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin
cd ..

# preparing ulm
git clone https://huggingface.co/soumi-maiti/speech-ulm-lstm
mv speech-ulm-lstm ulm
cd ..