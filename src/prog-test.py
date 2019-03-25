import sys
sys.path.append('../../virtualhome/simulation')

import resource
import os
import copy
import json
import numpy as np
from termcolor import colored
from multiprocessing import Pool

import torch

from helper import LCS, average_over_list
from program.utils import setup, summary_inference
from program.dataset import get_prog_dataset
from program.dataset import prog_collate_fn as collate_fn
from program.dataset import prog_to_cuda_fn as to_cuda_fn
from network.encoder_decoder import Sketch2Program


def test(args, model, test_loader, testing_dir):

    simulator_keys = ['parsibility', 'executability', 'final_state']
    f1_keys = ['attribute_precision', 'relation_precision', 'total_precision',
               'attribute_recall', 'relation_recall', 'total_recall',
               'attribute_f1', 'relation_f1', 'total_f1']

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
    print(colored('Start testing...', 'red'))
    print("Size of the testing set: {}".format(len(test_loader.dataset.programs)))
    model.eval()

    programs = []
    graphs = []
    feats = []
    dset = test_loader.dataset
    pool = Pool(processes=max(args.n_workers, 1))

    for batch_data in test_loader:

        batch_path_data = batch_data[3]
        _, program_predict_str, program_str, sketch_str, f1_results, \
                simulator_results, node_names = summary_inference(args, model, batch_data, dset, pool)

        for pred, program, sketch, simulator_result, f1_result, file_p, graph_p, node_name in zip(
                program_predict_str, program_str, sketch_str, simulator_results, f1_results, batch_path_data[0],
                batch_path_data[1], node_names):

            lcs = LCS(pred.split(', '), program.split(', '))
            lcs_score = len(lcs) / (float(max(len(pred.split(', ')), len(program.split(', ')))))
            p = {
                "file_path": file_p,
                "graph_path": graph_p,
                "given_sketch": sketch,
                "gt_prog": program,
                "pred_prog": pred,
                "lcs_gt_prog": list(_find_match(program.split(', '), copy.deepcopy(lcs))),
                "lcs_pred_prog": list(_find_match(pred.split(', '), copy.deepcopy(lcs)))}

            p.update({'lcs': lcs_score})
            for i, k in enumerate(f1_keys):
                p.update({k: f1_result[i]})

            for i, k in enumerate(simulator_keys):
                if k != 'final_state':
                    p.update({k: simulator_result[i]})
            graphs.append({
                'file_path': file_p,
                'all_states': simulator_result[-1]})

            feats.append({
                'file_path': file_p,
                'node_names': node_name})
            programs.append(p)


    pool.close()
    pool.join()

    data = {'checkpoint': args.checkpoint, 'programs': programs}
    if args.sketch_path is not None:
        data.update({'sketch_path': args.sketch_path})

    lcs_average = [p['lcs'] for p in programs]
    lcs_average = average_over_list(lcs_average)
    print('lcs: {}'.format(lcs_average))
    data.update({'lcs': lcs_average})

    for k in f1_keys:
        v = average_over_list([p[k] for p in programs])
        print("{}: {}".format(k, v))
        data.update({k: v})

    for k in simulator_keys:
        if k != 'final_state':
            v = average_over_list([p[k] for p in programs])
            print("{}: {}".format(k, v))
            data.update({k: v})

    # find which model is used
    model_iteration = args.checkpoint.split('/')[-1]
    model_iteration = model_iteration.replace('.ckpt', '')

    results_prefix = args.results_prefix if args.results_prefix else "testing_results-{}".format(
        model_iteration)
    results_prefix = os.path.join(testing_dir, results_prefix)

    json.dump(data, open('{}.json'.format(results_prefix), 'w'))
    print("File save in {}.json".format(results_prefix))


def main():

    # setup the config variables
    args, checkpoint_dir, _, model_config = setup(train=False)

    # get dataset
    _, test_dset = get_prog_dataset(args, train=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False)

    # loader for the testing set
    def _loader():
        while True:
            for batch_data in test_loader:
                yield batch_data

    # initialize model and optim
    model = Sketch2Program(test_dset, **model_config)
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
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
