# torch 2.8, cuda 12
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# install llama-factory
pip install -e ".[metrics]"
pip install -e ".[metrics,deepspeed]"

# vllm 0.11.0
pip install vllm==0.11.0

# flash-attn 2.8.3
pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation

# liger-kernel 0.6.4
pip install liger-kernel==0.6.4

# lmms-eval
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv pip install -e ".[all]"
cd ..