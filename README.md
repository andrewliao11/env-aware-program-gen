# Synthesizing Environment-Aware Activities via Activity Sketches

This is the official implementation of ResActGraph (CVPR2019).

[**Synthesizing Environment-Aware Activities via Activity Sketches**](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liao_Synthesizing_Environment-Aware_Activities_via_Activity_Sketches_CVPR_2019_paper.pdf)

*[Yuan-Hong Liao](https://andrewliao11.github.io)∗, [Xavier Puig](https://people.csail.mit.edu/xavierpuig/)∗, Marko Boben, [Antonio Torralba](http://web.mit.edu/torralba/www/), [Sanja Fidler](http://www.cs.utoronto.ca/~fidler/)*

If you find the code useful in your research, please consider citing:

```
@InProceedings{Liao_2019_CVPR,
author = {Liao, Yuan-Hong and Puig, Xavier and Boben, Marko and Torralba, Antonio and Fidler, Sanja},
title = {Synthesizing Environment-Aware Activities via Activity Sketches},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Contents 
- Introduction
- Environment Setup
- Training


## Introduction

<img align="right" src="asset/teaser.png" width="350" height="250">

In order to perform activities from demonstrations or descriptions,
agents need to distill what the essense of the given activity is. 
In this work, we address the problem of *environment-aware program generation*.
Given a visual demonstration or a description of an activity, 
we generate program sketches representing the essential instructions
and propose a model, **ResActGraph**, to transform these into full programs
representing the actions needed to perform the activity under the presented environmental constraints.

<br>

### Demo

Here is one short clip where the agent is chilling out in his living room.

<p align="center"><img src="asset/vh_intro.gif" width="600" height="400">


## Environment Setup

### Create a virtual environment (Optional)

```bash
$ virtualenv -p python3 virtualhome
$ source virtualhome/bin/activate
(virtualhome) $ git clone https://github.com/andrewliao11/env-aware-program-gen.git
(virtualhome) $ cd env-aware-program-gen
(virtualhome) $ pip3 install -r requirements.txt
```

### Install VirtualHome
To execute or evalutate the sampled programs, VirtualHome need to be installed.
Please see [here](https://github.com/xavierpuigf/virtualhome) for the installation.


### Data

Download the program dataset [here](http://virtual-home.org).

Here is how the dataset structure should look like:

```
dataset
└── VirtualHome-Env
    ├── augment_programs
    │   ├── augment_exception
    │   └── augment_location
    ├── demonstration
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
