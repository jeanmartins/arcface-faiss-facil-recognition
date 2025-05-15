## Setup

After installing [Conda](https://docs.conda.io/en/latest/miniconda.html), run the following commands in the root of the project:

```bash
conda create -n arcface-env python=3.10 -y
conda activate arcface-env

pip install -r requirements.txt
python main.py

```

Running on GPU (optional)
To enable GPU support, make sure your system has an NVIDIA GPU and the required drivers/libraries installed:

1. Install the CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

Recommended version: CUDA 12.x

2. Install cuDNN (Deep Neural Network library):
https://developer.nvidia.com/rdp/cudnn-download
