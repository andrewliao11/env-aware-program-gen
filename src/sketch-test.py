import resource
import os
import time
import copy
import json
import argparse
import random
import re
import numpy as np
from termcolor import colored
from multiprocessing import Pool

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from sketch.utils import setup
from helper import average_over_list, LCS

from sketch.utils import summary_inference


def test(args, model, test_loader, test_dir):

    def _find_match(full_program, lcs):
        idx = np.zeros(len(full_program))

        for i, atomic_action in enumerate(full_program):
            try:
                if atomic_action in lcs:
                    lcs.remove(atomic_action)
                    idx[i] = 1
            except IndexError:
                continue
        return idx

    # Test
    print('Start testing...')
    model.eval()

    programs = []
    dset = test_loader.dataset
    pool = Pool(processes=max(args.n_workers, 1))
    
    for batch_data in test_loader:

        if args.src == "desc":
            batch_file_path = batch_data[3]
        elif args.src == "demo":
            batch_file_path, batch_image_file_path = batch_data[2]
        else:
            raise ValueError

        _, sketch_predict_str, sketch_str, _ = summary_inference(model, batch_data, dset, pool)


        for i, (pred, sketch) in enumerate(zip(sketch_predict_str, sketch_str)):

            path = batch_file_path[i]
            path = re.search('results.*txt', path).group(0)
            path = path.replace('.txt', '')
            
            lcs = LCS(pred.split(', '), sketch.split(', '))
            lcs_score = len(
                lcs) / (float(max(len(pred.split(', ')), len(sketch.split(', ')))))
            p = {
                "path": path,
                "pred": pred,
                "sketch": sketch,
                "LCS_gt": list(
                    _find_match(
                        sketch.split(', '),
                        copy.deepcopy(lcs))),
                "LCS_pred": list(
                    _find_match(
                        pred.split(', '),
                        copy.deepcopy(lcs)))}
            p.update({'lcs': lcs_score})
            
            if args.src == "demo":
                img_path = batch_image_file_path[i]
                p.update({'img_path': img_path})

            programs.append(p)


    pool.close()
    pool.join()
    lcs_average = [p['lcs'] for p in programs]
    lcs_average = average_over_list(lcs_average)
    print("lcs: {}".format(lcs_average))

    data = {}
    data.update({"LCS": lcs_average})
    data.update({"programs": programs})

    # find which model is used
    model_iteration = args.checkpoint.split('/')[-1]
    model_iteration = model_iteration.replace('.ckpt', '')

    results_prefix = "testing_results-{}".format(model_iteration)
    results_prefix = os.path.join(test_dir, results_prefix)

    json.dump(data, open('{}.json'.format(results_prefix), 'w'))
    print("File save in {}.json".format(results_prefix))


def main():

    args, checkpoint_dir, _, model_config = setup(train=False)

    if args.src == 'desc':
        from sketch.desc_dataset import get_dataset
        from sketch.desc_dataset import collate_fn
        from sketch.desc_dataset import to_cuda_fn
    elif args.src == 'demo':
        from sketch.demo_dataset import get_dataset
        from sketch.demo_dataset import collate_fn
        from sketch.demo_dataset import to_cuda_fn

    # get data
    _, test_dset = get_dataset(args, train=False)

    test_loader = DataLoader(
        dataset=test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False)


    # initialize model
    if args.src == 'desc':
        from network.encoder_decoder import Desc2Sketch
        model = Desc2Sketch(test_dset, **model_config)
    elif args.src == 'demo':
        from network.encoder_decoder import Demo2Sketch
        model = Demo2Sketch(test_dset, **model_config)

    if args.gpu_id is not None:
        model.cuda()
        model.set_to_cuda_fn(to_cuda_fn)

    # main loop
    model.load(args.checkpoint, verbose=True)
    test(args, model, test_loader, checkpoint_dir)


# See: https://github.com/pytorch/pytorch/issues/973
# the worker might quit loading since it takes too long
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024 * 4, rlimit[1]))

if __name__ == '__main__':
    # See: https://github.com/pytorch/pytorch/issues/1355
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
