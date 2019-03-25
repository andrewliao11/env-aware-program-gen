# Synthesizing Environment-Aware Activities via Activity Sketches

Here, we provide the implementation of `resactgraph` and some baselines, nearest neighbor, unary, and graph that we used in the CVPR paper.

The achitectural overview of `resactgraph`:

<p align="center"><img src="/asset/model.png"  height="300"></p>

## Program generation

- Run `nearest neighbors` baseline

```bash
(virtualhome) $ python prog-NN.py
```

Note: Files called `src/close_ids.npz` and `src/.sketch_and_prog_train.npz` will be dump after running the nearest neighbor and you can visualize the results using `viz_NN.py` by `python viz_NN.py close_ids.npz`. An html file will be generated and let you better check the results of NN.

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

We provide the pre-trained weights for `resactgraph` [here]() (xxx MB).


### Evaluate the model

To evalute the trained model (take `resactgraph` for example),

```bash
(virtualhome) $ python prog-test.py --graph_evolve True --n_timesteps 2 --model_type residual --checkpoint {PATH_TO_WEIGHT} --gpu_id 0 
```

After evaluation, it will dump a file `testing_results-sketch2program-xxx.json` that contains all the results.

Note: Program is one kind of formal language, so we can apply a simple grammar constraints during inference. To apply it, just add an additional flag `--use_constraint_mask True`


### Results

- The results of `resactgrapah` and other baselines

|                   | LCS | F1-state | F1-attribute | F1 |
|------------------ |-----|----------|--------------|----|
| Nearest Neighbour |     |          |              |    |
| Unary             |     |          |              |    |
| Graph             |     |          |              |    |
| ResActGraph       |     |          |              |    |


- The improvement after applying the constraint mask

|                              | LCS | F1-state | F1-attribute | F1 |
|------------------------------|-----|----------|--------------|----|
| Graph (w/ constraints)       |     |          |              |    |
| ResActGraph (w/ constraints) |     |          |              |    |


## Sketch generation

- Train `demo2sketch` from demonstrations

```bash
(virtualhome) $ python sketch-train.py --src demo --batch_size 16 --gpu_id 0 
```

- Train `desc2sketch` from descriptions

```bash
(virtualhome) $ python sketch-train.py --src desc --batch_size 128 --gpu_id 0 
```

Run `sketch_test.py` and the output file will be dumped in a json file.
We provide the output json file from at `src/sketch_prediction`

## Combine them together

- Test `sketch2program` with sketches predicted by `demo2sketch`

Add the flag `--sketch_path` to specify the the path to the sketch prediction (by default it's called `testing_results-xxxx2sketch-xxx.json`)


```bash
(virtualhome) $ python prog-test.py --sketch_path {PATH_TO_JSON} --checkpoint {PATH_TO_WEIGHT}  --gpu_id 0 
```
