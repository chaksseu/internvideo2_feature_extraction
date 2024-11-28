#apt-get update
#apt-get upgrade -y
#apt-get install -y curl wget git tar zip unzip libjpeg-dev libpng-dev libgl1-mesa-glx libglib2.0-0
#conda update -n base -c defaults conda -y
#bash
#conda init
#conda create -n videomae python=3.8 -y
#conda activate videomae
pip install --upgrade pip
cd VideoMAEv2
pip install -r requirements.txt
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
cd flash-attention
cd csrc/fused_dense_lib && pip install .
cd ../..
cd csrc/layer_norm && pip install .


#export PATH=$PATH:~/.local/bin
#accelerate config default
#huggingface-cli login
#
#MAX_JOBS=4 pip install flash-attn==2.0.8 --no-build-isolation