# Models

Here, we will split the model into two parts, [program generation](https://github.com/andrewliao11/env-aware-program-gen/tree/master/src#program-generation) and [sketch generation](https://github.com/andrewliao11/env-aware-program-gen/tree/master/src#sketch-generation). For program generation we provide the implementation of **ResActgraph** and some baselines in the paper. Also, all the trained weights can be found at [here](https://drive.google.com/open?id=1d5Y1f7tAqO4Ak70RvERusWEYWHIgneJv)

The achitectural overview of ResActGraph:

<p align="center"><img src="/asset/model.png"  height="300"></p>

## Program generation

- Run `nearest neighbors` baseline

```bash
(virtualhome) $ python prog-NN.py
```

Note: Files called `src/close_ids.npz` and `src/.sketch_and_prog_train.npz` will be dumped after running the nearest neighbor and you can visualize the results using `viz_NN.py` by `python viz_NN.py close_ids.npz`. An html file will be generated and let you better check the results of NN.

### Training

- Train `unary`, `graph`, `fcactgraph`, `gruactgraph`, and `resactgraph`

```bash
# unary
(virtualhome) $ python prog-train.py --prefix unary --graph_evolve False --n_timesteps 0 --gpu_id 0
# graph
(virtualhome) $ python prog-train.py --prefix graph --graph_evolve False --n_timesteps 2 --gpu_id 0
# fcactgraph
(virtualhome) $ python prog-train.py --prefix fcactgraph --graph_evolve True --n_timesteps 2 --model_type fc --gpu_id 0
# gruactgraph
(virtualhome) $ python prog-train.py --prefix gruactgraph --graph_evolve True --n_timesteps 2 --model_type gru --gpu_id 0
# resactgraph
(virtualhome) $ python prog-train.py --prefix resactgraph --graph_evolve True --n_timesteps 2 --model_type residual --gpu_id 0
```

The tensorboard files and checkpoints will be saved into `src/checkpoint_dir`

We provide the trained weights of `resactgraph` at [here](https://drive.google.com/open?id=1g027GxdM4lK9xAKPpEbzaeQd-SVs6gVk).


### Evaluating

To evalute the trained model (take `resactgraph` for example),

```bash
(virtualhome) $ python prog-test.py --graph_evolve True --n_timesteps 2 --model_type residual --checkpoint {PATH_TO_WEIGHT} --gpu_id 0 
```

After evaluation, it will dump a file `testing_results-sketch2program-xxx.json` that contains all the results.

Note: Program is one kind of formal language, so we can apply a simple grammar constraints during inference. To apply it, just add an additional flag `--use_constraint_mask True`


### Results

- Trained model evaluated with different evaluation metrics

|                   | LCS | F1-relation | F1-attribute | F1  |parsibility|executability|
|------------------ |-----|-------------|--------------|-----|-----------|-------------|
| Nearest Neighbour |0.127|     0.019   |     0.288    |0.041|     -     |       -     |
| Unary             |0.372|     0.160   |     0.142    |0.159|  0.753    |     0.248   |
| Graph             |0.400|     0.171   |     0.171    |0.172|  0.822    |     0.231   |
| FC                |0.469|     0.261   |     0.273    |0.263|  0.886    |     0.337   |
| GRU               |0.508|     0.410   |     0.408    |0.411|  0.879    |     0.489   |
| ResActGraph       |0.516|     0.410   |     0.420    |0.413|  0.853    |     0.493   |


- The improvement after applying the grammar constraint during inference

|                              | LCS | F1-relation | F1-attribute | F1  |parsibility|executability|
|------------------------------|-----|-------------|--------------|-----|-----------|-------------|
| Unary (w/ constraint)        |0.356|    0.124    |     0.125    |0.124|    0.675  |   0.159     |
| Graph (w/ constraint)        |0.401|    0.189    |     0.187    |0.186|    0.975  |   0.261     |
| FC (w/ constraint)           |0.475|    0.297    |     0.310    |0.299|    0.963  |   0.389     |
| GRU (w/ constraint)          |0.513|    0.468    |     0.469    |0.470|    0.953  |   0.578     |
| ResActGraph (w/ constraint)  |0.521|    0.443    |     0.456    |0.446|    0.970  |   0.554     |


## Sketch generation

### Training

- Train `demo2sketch` from demonstrations

```bash
(virtualhome) $ python sketch-train.py --src demo --batch_size 16 --gpu_id 0 
```

- Train `desc2sketch` from descriptions

```bash
(virtualhome) $ python sketch-train.py --src desc --batch_size 128 --gpu_id 0 
```

Run `sketch_test.py` and the output file will be dumped in a json file.
We provide the output json file from at `src/sketch_prediction` and the trained weight at [here](https://drive.google.com/open?id=1eHbm9v9j0JewChayldvOuUcZkVHkN-p3) and [here](https://drive.google.com/open?id=1eFqLe4nJ10SdzFwHI8xrvzYtJPF6XVs7).

## Sketch generation + Program generation

Our final goal is to generate programs from descriptions or demonstrations. To this end, we directly combine the above two parts.

- Generate program from descriptions or demonstrations

Add the flag `--sketch_path` to specify the the path to the sketch prediction (by default it's called `testing_results-xxxx2sketch-xxx.json`). We provide the sketch prediction in `src/sketch_prediction`


```bash
(virtualhome) $ python prog-test.py --sketch_path {PATH_TO_JSON} --checkpoint {PATH_TO_WEIGHT}  --gpu_id 0 
```


### Results

- Generate program from descriptions

|                   | LCS | F1-relation | F1-attribute | F1  |parsibility|executability|
|------------------ |-----|-------------|--------------|-----|-----------|-------------|
| Unary             |0.289|     0.188   |     0.191    |0.189|    0.734  |   0.444     |
| Graph             |0.297|     0.24    |     0.233    |0.241|    0.899  |   0.439     |
| ResActGraph       |0.331|     0.347   |     0.339    |0.348|    0.925  |   0.633     |


- Generate program from demonstrations

|                   | LCS | F1-relation | F1-attribute | F1  |parsibility|executability|
|------------------ |-----|-------------|--------------|-----|-----------|-------------|
| Unary             |0.257|     0.099   |     0.09     |0.098|    0.74   |   0.256     |
| Graph             |0.284|     0.202   |     0.202    |0.203|    0.828  |   0.343     |
| ResActGraph       |0.327|     0.316   |     0.323    |0.318|    0.860  |   0.547     |


## Other: some concerns about the DataLoader

- If you encounter `/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown len(cache))`

:point_right: check the issue [here](https://github.com/pytorch/pytorch/issues/11727)

- If you encounter `RuntimeError: unable to open shared memory object </torch_29919_1396182366> in read-write mode`

:point_right: check the issue [here](https://github.com/pytorch/pytorch/issues/13246) and [here](https://github.com/facebookresearch/maskrcnn-benchmark/issues/103)
