## [ICLR 2023 Harnessing Out-Of-Distribution Examples via Augmenting Content and Style](https://openreview.net/pdf?id=boNyg20-JDm)

Zhuo Huang<sup>1</sup>, Xiaobo Xia<sup>1</sup>, Li Shen<sup>2</sup>, Bo Han<sup>3</sup>, Mingming Gong<sup>4</sup>, Chen Gong<sup>5</sup>, Tongliang Liu<sup>1</sup>

<sup>1</sup>The University of Sydney, <sup>2</sup>JD Explore Academy, <sup>3</sup>Hong Kong Baptist University, <sup>4</sup>University of Melbourne, <sup>5</sup>Nanjing University of Science and Technology

![HOOD Overview](images/scm.png)

## Abstract
Machine learning models are vulnerable to Out-Of-Distribution (OOD) examples, and such a problem has drawn much attention. However, current methods lack a full understanding of different types of OOD data: there are benign OOD data that can be properly adapted to enhance the learning performance, while other malign OOD data would severely degenerate the classification result. To Harness OOD data, this paper proposes a HOOD method that can leverage the content and style from each image instance to identify benign and malign OOD data. Particularly, we design a variational inference framework to causally disentangle content and style features by constructing a structural causal model. Subsequently, we augment the content and style through an intervention process to produce malign and benign OOD data, respectively. The benign OOD data contain novel styles but hold our interested contents, and they can be leveraged to help train a style-invariant model. In contrast, the malign OOD data inherit unknown contents but carry familiar styles, by detecting them can improve model robustness against deceiving anomalies. Thanks to the proposed novel disentanglement and data augmentation techniques, HOOD can effectively deal with OOD examples in unknown and open environments, whose effectiveness is empirically validated in three typical OOD applications including OOD detection, open-set semi-supervised learning, and open-set domain adaptation.

This is an PyTorch implementation of HOOD.
This implementation is based on [Pytorch-FixMatch](https://github.com/kekmodel/FixMatch-pytorch) and [OpenMatch](https://github.com/VisionLearningGroup/OP_Match).


## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- sklearn
- apex (optional)

See [Pytorch-FixMatch](https://github.com/kekmodel/FixMatch-pytorch) for the details.

## Usage

### Dataset Preparation
This repository needs CIFAR10, CIFAR100, or ImageNet-30 to train a model.

To conduct OOD detection, we also need SVHN, LSUN, ImageNet
for CIFAR10, 100, and LSUN, DTD, CUB, Flowers, Caltech_256, Stanford Dogs for ImageNet-30.
To prepare the datasets above, follow [CSI](https://github.com/alinlab/CSI).


```
mkdir data
ln -s path_to_each_dataset ./data/.

## unzip filelist for imagenet_30 experiments.
unzip files.zip
```

All datasets are supposed to be under ./data.

### Train
Train the model by 50 labeled data per class of CIFAR-10 dataset:

```
sh run_cifar10.sh 50 save_directory
```

Train the model by 50 labeled data per class of CIFAR-100 dataset, 55 known classes:

```
sh run_cifar100.sh 50 10 save_directory
```


Train the model by 50 labeled data per class of CIFAR-100 dataset, 80 known classes:

```
sh run_cifar100.sh 50 15 save_directory
```


Run experiments on ImageNet-30:

```
sh run_imagenet.sh save_directory
```


### Evaluation
Evaluate a model trained on cifar10

```
sh run_eval_cifar10.sh trained_model.pth
```


### Acknowledgement
This repository depends a lot on [Pytorch-FixMatch](https://github.com/kekmodel/FixMatch-pytorch) for FixMatch implementation, and [CSI](https://github.com/alinlab/CSI) for anomaly detection evaluation, and [OpenMatch](https://github.com/VisionLearningGroup/OP_Match) for Open Classifier implementation. 
 Appreciate their contributions.

### Reference
If you find this code helpful, please consider citing our paper, thanks.

```
@inproceedings{huang2023harnessing,
  title={Harnessing Out-Of-Distribution Examples via Augmenting Content and Style},
  author={Zhuo Huang and Xiaobo Xia and Li Shen and Bo Han and Mingming Gong and Chen Gong and Tongliang Liu},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=boNyg20-JDm}
}
```

