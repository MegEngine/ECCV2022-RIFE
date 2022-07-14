# Real-Time Intermediate Flow Estimation for Video Frame Interpolation (MegEngine implementation)

## [YouTube](https://www.youtube.com/results?search_query=rife+interpolation&sp=CAM%253D) | [BiliBili](https://search.bilibili.com/all?keyword=SVFI&order=stow&duration=0&tids_1=0) | [Tutorial](https://www.youtube.com/watch?v=gf_on-dbwyU&feature=emb_title)
16X interpolation results from two input images: 

![Demo](./demo/I2_slomo_clipped.gif)
![Demo](./demo/D2_slomo_clipped.gif)



## Introduction
This project is an official MegEngine implementation of [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294). For Pytorch implementation, please refers to [this repo](https://github.com/megvii-research/ECCV2022-RIFE). Currently, our model can run 30+FPS for 2X 720p interpolation on a 2080Ti GPU. It supports arbitrary-timestep interpolation between a pair of images. 

## CLI Usage

### Installation

```
git clone git@github.com:MegEngine/ECCV2022-RIFE-MegEngine
cd ECCV2022-RIFE-MegEngine
pip3 install -r requirements.txt
```

* Download the pretrained **HD** models from [here](https://data.megengine.org.cn/research/rife/flownet/flownet.pkl).
* Unzip and move the pretrained parameters to train_log/\*
* This model is not reported by our paper, for our paper model please refer to [evaluation](https://github.com/MegEngine/ECCV2022-RIFE-MegEngine/#evaluation).

### Run

**Image Interpolation**

```
python3 inference_img.py --img img0.png img1.png --exp=4
```
(2^4=16X interpolation results)
After that, you can use pngs to generate mp4:

```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -c:v libx264 -pix_fmt yuv420p output/slomo.mp4 -q:v 0 -q:a 0
```
You can also use pngs to generate gif:
```
ffmpeg -r 10 -f image2 -i output/img%d.png -s 448x256 -vf "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1" output/slomo.gif
```

## Evaluation
Download [RIFE model](https://data.megengine.org.cn/research/rife/flownet/flownet.pkl) or [RIFE_m model](https://data.megengine.org.cn/research/rife/flownet_m/flownet.pkl) reported by our paper.

**MiddleBury**: Download [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/) at ./other-data and ./other-gt-interp

**HD**: Download [HD dataset](https://github.com/baowenbo/MEMC-Net) at ./HD_dataset. We also provide a [google drive download link](https://drive.google.com/file/d/1iHaLoR2g1-FLgr9MEv51NH_KQYMYz-FA/view?usp=sharing).

We provide code for evaluating with datasets above, please follow lines:

```bash
python3 benchmark/HD_multi_4X.py
python3 benchmark/HD.py
python3 benchmark/MiddleBury_Other.py
python3 benchmark/yuv_frame_io.py
python3 testtime.py
```

## Training and Reproduction
Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

We use 16 CPUs, 4 GPUs and 20G memory for training: 
```
python3 train.py --arbitrary=False
```

## Citation

```
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Reference

Optical Flow:
[ARFlow](https://github.com/lliuz/ARFlow)  [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet)  [RAFT](https://github.com/princeton-vl/RAFT)  [pytorch-PWCNet](https://github.com/sniklaus/pytorch-pwc)

Video Interpolation: 
[DVF](https://github.com/lxx1991/pytorch-voxel-flow)  [TOflow](https://github.com/Coldog2333/pytoflow)  [SepConv](https://github.com/sniklaus/sepconv-slomo)  [DAIN](https://github.com/baowenbo/DAIN)  [CAIN](https://github.com/myungsub/CAIN)  [MEMC-Net](https://github.com/baowenbo/MEMC-Net)   [SoftSplat](https://github.com/sniklaus/softmax-splatting)  [BMBC](https://github.com/JunHeum/BMBC)  [EDSC](https://github.com/Xianhang/EDSC-pytorch)

