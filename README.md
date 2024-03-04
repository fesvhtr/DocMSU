# DocMSU: A Comprehensive Benchmark for Document-level Multimodal Sarcasm Understanding
Hang Du, Guoshun Nan, Sicheng Zhang, Binzhu Xie, Junrui Xu, Hehe Fan, Qimei Cui, Xiaofeng Tao, Xudong Jiang
This repo is the official dataset and Pytorch implementation of [DocMSU: A Comprehensive Benchmark for Document-level Multimodal Sarcasm Understanding](https://arxiv.org/abs/2312.16023) [AAAI2024].

## Introducing DocMSU
## DocMSU Dataset
Please download the dataset from [here](). There are two files: `img.zip`, `anno.zip`(Images and annotation files).  
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
Download checkpoint for swin-transformer [here]().  
Download recommended checkpoint for DocMSU [here]().
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
```
