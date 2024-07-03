# SUGAR

SUGAR: Spherical ultrafast graph attention framework for cortical surface registration

## Install dependencies

### FreeSurfer

Install FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

### Python

python==3.10.*

```shell
conda create -n SUGAR python==3.10.14 pip
conda activate SUGAR
```

packages

```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install fvcore

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install nibabel pytorch_warmup
```

### Run

Please set the path in train.py first.

```shell
cd SUGAR
python3 train.py
```
