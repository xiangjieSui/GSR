# Perceptual Quality Assessment for Omnidirectional Images by Generative Scanpath Representation
Xiangjie Sui*, Hanwei Zhu*, Xuelin Liu, Yuming Fang, Shiqi Wang, and Zhou Wang  
  
[:sparkles:Paper](https://arxiv.org/abs/2309.03472)

$\color{red}{We\ are\ cleaning\ up\ the\ code}$

# Setting Up  
* __Enviorment__  
  ```
  #  Pytorch 2.0.1+cu117 & CUDA 11.7 
  conda env create -f requirements.yaml
  ```
* __Datasets__  
  We test three public datasets [CVIQ](https://github.com/sunwei925/CVIQDatabase), [OIQA](https://mega.nz/file/FqxxRQRR#4Ju2qcmmo6Ced_7nRBXXqAaDcjqxjH2uUFnXIeyE2ts), [JUFE](https://github.com/LXLHXL123/JUFE-VRIQA). We organize the files as:
  ```
  -imgs
      -img1
      -img2
      -...
  -mos.pkl  # a set of hash indexes: img_name -> mos 
  ```
* __Pre-trained Models__    
   We test three 3D backbone: Swin-T, ConvNet, and Xclip. The scanpath generator is derived from our CVPR 2023 paper ([Paper](https://ece.uwaterloo.ca/~z70wang/publications/CVPR23_scanPath360Image.pdf) & [Code](https://github.com/xiangjieSui/ScanDMM)) You can download these pre-trained models from corresponding authors. Alternatively, you can also download them from the sources we provided [[Google Drive]](https://drive.google.com/drive/folders/1Mw3Ep4FJU8G0Ft9DCgPBj1LLFesMUnhU?usp=drive_link).
   ```
   cd ./model
   ls
   convnext_tiny_1k_224_ema.pth,
   swin_tiny_patch244_window877_kinetics400_1k.pth,
   k400_32_16.pth
   scandmm-seed-1238.pkl
   ```
# Training  
* __Runing Command__  
```
# JUFE
python -u train.py --db='./Dataset/JUFE' --nw=3 --backbone='xclip' --dbsd=1234 --bs=16 --lr=8e-6

# CVIQ
python -u train.py --db='./Dataset/CVIQ' --nw=3 --backbone='xclip' --dbsd=1234 --bs=16 --lr=8e-7

# OIQA
python -u train.py --db='./Dataset/OIQA' --nw=3 --backbone='xclip' --dbsd=1234 --bs=8 --lr=8e-7
```
* __Check the Checkpoint and Final Model__  
```
cd ./checkpoints
ls
JUFE-X-seed-1238.pth

cd ./model
ls
JUFE-X-seed-1238 
```
# Test  
We provided some pre-trained models [[here]](https://drive.google.com/drive/folders/1djA83UB5bcf-ue5YvW6CUa5e9A_-KE20?usp=drive_link).
```
cd ./model
ls
scandmm-seed-1238.pkl
CVIQ-X-seed-1238,
OIQA-X-seed-1238,
JUFE-X-seed-1238

# test
python -u train.py --test=True --cp=True --db='./Dataset/JUFE' --backbone='xclip' --dbsd=1238 
```

# Bibtex
```
@article{gsr2023,
  title={Perceptual Quality Assessment of 360° Images Based on Generative Scanpath Representation},
  author={Xiangjie Sui and Hanwei Zhu and Xuelin Liu and Yuming Fang and Shiqi Wang and Zhou Wang},
  year={2023},
  eprint={2309.03472},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
