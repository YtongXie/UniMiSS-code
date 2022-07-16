# UniMiSS-code
This is the official pytorch implementation of our ECCV 2022 paper "UniMiSS: Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier". In this paper, we advocate bringing a wealth of 2D images like chest X-rays as compensation for the lack of 3D data, aiming to build a universal medical self-supervised representation learning framework, called UniMiSS. We conduct expensive experiments on six 3D/2D medical image analysis tasks, including segmentation and classification. The results show that the proposed UniMiSS achieves promising performance on various downstream tasks, outperforming the ImageNet pre-training and other advanced SSL counterparts substantially.

<div align="center">
  <img width="100%" alt="DINO illustration" src=".github/Fig2.pdf">
</div>













```
@article{UniMiSS,
  title={UniMiSS: Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier},
  author={Xie, Yutong and Zhang, Jianpeng and Xia, Yong and Wu, Qi},
  booktitle={ECCV},
  year={2022}
}
  
```

### 5. Acknowledgements
Part of codes is reused from the [DINO](https://github.com/facebookresearch/dino). Thanks to Caron et al. for the codes of DINO.

### Contact
Yutong Xie (yutong.xie678@gmail.com)
