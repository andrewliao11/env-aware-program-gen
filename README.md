# Synthesizing Environment-Aware Activities via Activity Sketches

This is the official implementation of ResActGraph (CVPR2019). For technical details, please refer to [here](ARXIV_LINK)

**Synthesizing Environment-Aware Activities via Activity Sketches**

*[Yuan-Hong Liao](https://andrewliao11.github.io)∗, Xavier Puig∗, Marko Boben, Antonio Torralba, Sanja Fidler*

If you find the code useful in your research, please consider citing:

```
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
    author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
}
```

## Contents 
- Introduction
- Environment Setup
- Training


## Introduction

In order to perform activities from demonstrations or descriptions,
agents need to distill what the essense of the given activity is. 
In this work, we address the problem of environment-aware program generation.
Given a visual demonstration or a description of an activity, 
we generate program sketches representing the essential instructions
and propose a model, `ResActGraph`, to transform these into full programs
representing the actions needed to perform the activity under the presented environmental constraints.

<p align="center"><img src="asset/vh_intro.gif" width="450" height="300"><img src="asset/teaser.png" width="350" height="300"></p>


## Environment Setup

### Create a virtual environment (Skip if you already have)

```bash
$ virtualenv -p python3 virtualhome
$ source virtualhome/bin/activate
(virtualhome) $ git clone https://github.com/andrewliao11/env-aware-program-gen.git
(virtualhome) $ cd env-aware-program-gen
(virtualhome) $ pip3 install -r requirements.txt
```

### Install VirtualHome
To execute or evalutate the sampled programs, VirtualHome need to be installed.
Please see [here]() for the installation.


### Dataset structure

Download the program dataset [here](xxx) and the augmented program dataset [here](xxx)

Here is how the dataset structure should look like:

```
dataset
└── VirtualHome-Env
    ├── augment_programs
    │   ├── augment_exception
    │   └── augment_location
    ├── demos
    │   ├── images
    │   └── images_augment
    ├── original_programs
    ├── resources
    │   ├── class_name_equivalence.json
    │   ├── knowledge_base.npz
    │   ├── object_merged.json
    │   ├── object_prefabs.json
    │   └── object_script_placing.json
    ├── sketch_annotation.json
    └── split
        ├── test_progs_paths.txt
        └── train_progs_paths.txt
```

## Training

The training of the program/sketch generation model is documneted [here](/src/README.md)
