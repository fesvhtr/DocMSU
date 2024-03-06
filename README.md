# DocMSU: A Comprehensive Benchmark for Document-level Multimodal Sarcasm Understanding
This repo is the official dataset and Pytorch implementation of [DocMSU: A Comprehensive Benchmark for Document-level Multimodal Sarcasm Understanding](https://arxiv.org/abs/2312.16023) [AAAI2024].  
**Maintaining - We will complete the repo within a week.**
## Introducing DocMSU
## DocMSU Dataset
Please download the dataset from [here](https://drive.google.com/drive/folders/1g4jI9ZVGtNd3pXm7y7cZkimDur5u50Fq?usp=sharing). Here are two files: `img.zip`, `anno.zip` (Images and annotation files).  
Put them into `./DocMSU/data/release/` and unzip all.
## Get Started
```
git clone https://github.com/fesvhtr/DocMSU.git
cd DocMSU
conda create -n docmsu python=3.8
pip install -r requirements.txt
conda activate docmsu
```
## Checkpoints
Download checkpoint `swin_base_patch4_window7_224.pth` `swin_small_patch4_window7_224.pth` `swin_tiny_patch4_window7_224.pth` for swin-transformer [here](https://github.com/microsoft/Swin-Transformer).  
Download recommended `textmodel_8.pth` `visualmodel_8.pth` checkpoint for DocMSU [here]().
## Cite
```
@misc{du2023docmsu,
      title={DocMSU: A Comprehensive Benchmark for Document-level Multimodal Sarcasm Understanding}, 
      author={Hang Du and Guoshun Nan and Sicheng Zhang and Binzhu Xie and Junrui Xu and Hehe Fan and Qimei Cui and Xiaofeng Tao and Xudong Jiang},
      year={2023},
      eprint={2312.16023},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}