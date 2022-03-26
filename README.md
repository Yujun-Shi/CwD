# (CVPR 2022) Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning [ArXiv](https://arxiv.org/abs/2112.04731)
This repo contains Official Implementation of our CVPR 2022 paper: Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning.



### Abstract

Class Incremental Learning (CIL) aims at learning a classifier in a phase-by-phase manner, in which only data of a subset of the classes are provided at each phase. Previous works mainly focus on mitigating forgetting in phases after the initial one. However, we find that improving CIL at its initial phase is also a promising direction. Specifically, we experimentally show that directly encouraging CIL Learner at the initial phase to output similar representations as the model jointly trained on all classes can greatly boost the CIL performance. Motivated by this, we study the difference between a na\"ively-trained initial-phase model and the oracle model. Specifically, since one major difference between these two models is the number of training classes, we investigate how such difference affects the model representations. We find that, with fewer training classes, the data representations of each class lie in a long and narrow region; with more training classes, the representations of each class scatter more uniformly. Inspired by this observation, we propose **C**lass-**w**ise **D**ecorrelation (**CwD**) that effectively regularizes representations of each class to scatter more uniformly, thus mimicking the model jointly trained with all classes (i.e., the oracle model). Our CwD is simple to implement and easy to plug into existing methods. Extensive experiments on various benchmark datasets show that CwD consistently and significantly improves the performance of existing state-of-the-art methods by around 1% to 3%.

<\br><\br><\br>



### Instructions to Run Our Code

Current codebase only contain experiments on [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) with CIFAR100 and ImageNet100. Code reproducing results based on [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) and [AANet](https://github.com/yaoyao-liu/class-incremental-learning) are based on their repo and will be coming soon!

<\br>

#### CIFAR100 Experiments w/ LUCIR

No need to download the datasets, everything will be dealt with automatically.

For LUCIR baseline, simply first navigate under "src" folder and run:

```bash
bash exp_cifar_lucir.sh
```

For LUCIR + CwD, first navigate under "src" folder and run:

```bash
bash exp_cifar_lucir_cwd.sh
```

#### ImageNet100 Experiments w/ LUCIR

To run ImageNet100, please follow the following two steps:

Step 1:

download and extract imagenet dataset under "src/data/imagenet" folder.

Then, under "src/data/imagenet", run:

```bash
python3 gen_lst.py
```

 This command will generate two list that determine the order of classes for class incremental learning. The class order is shuffled by seed 1993 like most previous works.

<\br>

Step 2:

For LUCIR baseline, first navigate under "src" folder and run:

```bash
bash exp_im100_lucir.sh
```

For LUCIR+CWD, first navigate under "src" folder and run:

```bash
bash exp_im100_lucir_cwd.sh
```



#### Some Comments on Running Scripts.

For "SEED" variable in the scripts, it is not the seed that used to shuffle the class order, it is the seed that determines model initialisation/data loader sampling, etc. We vary "SEED" from 0,1,2 and average the Average Incremental Accuracy to obtain results reported in the paper.

<\br><\br><\br>



### For customized usage

To use our CwD loss in your own project, simply copy and paste the CwD loss implemented in "src/approach/aux\_loss.py" will be fine.

<\br><\br><\br>



### Citation

If you find our repo/paper helpful, please consider citing our work :)
```
@article{shi2021mimicking,
  title={Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning},
  author={Shi, Yujun and Zhou, Kuangqi and Liang, Jian and Jiang, Zihang and Feng, Jiashi and Torr, Philip and Bai, Song and Tan, Vincent YF},
  journal={arXiv preprint arXiv:2112.04731},
  year={2021}
}
```



### Contact

Yujun Shi (shi.yujun@u.nus.edu)



### Acknowledgements

Our code is based on [FACIL](https://github.com/mmasana/FACIL), one of the most well-written CIL library in my opinion:)



### Some Additional Remarks

Based on the original implementation of FACIL, I also implemented Distributed Data Parallel to enable multi-GPU training. However, it seems that the performance is not as good as single card training (about 0.5% lower). Therefore, in all experiments, I still use single card training.
