import sys
sys.path.append('../../virtualhome/simulation')

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from program.dataset import get_prog_dataset
from program.utils import grab_args, evaluate_f1_scores, LCS


def obtain_lcs_sketch(sketch_str_test, sketch_str_train):
    lcs = LCS(sketch_str_train.split(', '), sketch_str_test.split(', '))
    lcs_s = len(lcs) / (float(max(len(sketch_str_test.split(', ')),
                                  len(sketch_str_train.split(', ')))))
    return lcs_s


def get_programs_and_sketches(train_dset):

    filename = '.sketch_and_prog_train'
    if not os.path.isfile(filename + '.npz'):
        sketch_and_prog_train = []

        for data_train in tqdm(train_dset):
            prog_train, sketch_train, graph_train, _ = data_train
            sketch_train_b = [x.unsqueeze(0) for x in sketch_train[1:]]
            sketch_train_b[-1] *= 0
            sketch_train_b[-2] *= 0
            sketch_str_train = train_dset.interpret_sketch(*sketch_train_b)[0]
            progr_train_b = [x.unsqueeze(0) for x in list(prog_train)[1:]]
            node_name_list_tr = graph_train[1]
            progr_train_b.append([node_name_list_tr])
            prog_train_str = train_dset.interpret(*progr_train_b)[0]
            sketch_and_prog_train.append((sketch_str_train, prog_train_str))
            
        np.savez(filename, sketch_and_prog_train=sketch_and_prog_train)
    else:
        print("Load file: {}".format(filename + '.npz'))
        file = np.load(filename + '.npz')
        sketch_and_prog_train = file['sketch_and_prog_train']

    return sketch_and_prog_train


def main():

    # setup the config variables
    args = grab_args()

    # get dataset
    train_dset, test_dset = get_prog_dataset(args, train=True)
    close_ids, scores, prog_t, prog_closest, distances = [], [], [], [], []

    # Save the sketch and programs from the training set
    sketch_and_prog_train = get_programs_and_sketches(train_dset)
    pool = Pool(processes=args.n_workers)

    for data_test in tqdm(test_dset):

        prog_test, sketch_test, graph_test, _ = data_test
        graph_test_init = graph_test[-2]
        graph_test_end = graph_test[-1]

        sketch_test_b = [x.unsqueeze(0) for x in sketch_test[1:]]
        sketch_test_b[-1] *= 0
        sketch_test_b[-2] *= 0
        sketch_str_test = test_dset.interpret_sketch(*sketch_test_b)[0]
        progr_test_b = [x.unsqueeze(0) for x in list(prog_test)[1:]]
        node_name_list_test = graph_test[1]
        progr_test_b.append([node_name_list_test])
        prog_test_str = test_dset.interpret(*progr_test_b)[0]

        # Compute all lcs scores
        sketch_str_train = [sketch_and_prog_train[train_id][0]
                            for train_id in range(len(train_dset))]
        sketch_str_test = [sketch_str_test] * len(train_dset)
        lcs_scores = [
            obtain_lcs_sketch(
                x, y) for x, y in zip(
                sketch_str_train, sketch_str_test)]

        # Get best LCS
        best_lcs_score = np.max(lcs_scores)
        ids = np.where(lcs_scores == best_lcs_score)[0]
        if best_lcs_score == 0:
            ids = [ids[0]]

        # Compute F scores for that LCS
        input_f = []
        for id_train in ids:
            prog_train_str = sketch_and_prog_train[id_train][1]
            _, _, graph_train, _ = train_dset[id_train]
            graph_tr = graph_train[-2]
            input_f.append([prog_test_str, graph_test_init,
                            prog_train_str, graph_tr])

        scores_graph = pool.map(evaluate_f1_scores, input_f)
        scores_graph = [x[-1] for x in scores_graph]

        max_score_graph = np.max(scores_graph)
        max_train_id = ids[np.argmax(scores_graph)]
        prog_trainclosest_str = sketch_and_prog_train[max_train_id][1]

        prog_closest.append(prog_trainclosest_str)
        prog_t.append(prog_test_str)
        close_ids.append(max_train_id)
        distances.append((best_lcs_score, max_score_graph))

        graph_train_end = train_dset[max_train_id][-2][-1]
        scores.append({
            'LCS': obtain_lcs_sketch(prog_trainclosest_str, prog_test_str),
            'F1': evaluate_f1_scores([prog_trainclosest_str, graph_train_end, prog_test_str, graph_test_end])})

    np.savez(
        'close_ids',
        close_ids=close_ids,
        scores=scores,
        distances=distances,
        prog_test=prog_t,
        prog_train=prog_closest)

    mean_LCS = np.mean([x['LCS'] for x in scores])
    keys = [
        'attribute_precision',
        'relation_precision',
        'total_precision',
        'attribute_recall',
        'relation_recall',
        'total_recall',
        'attribute_f1',
        'relation_f1',
        'total_f1']

    print('Mean LCS: {}'.format(mean_LCS))
    for it, key in enumerate(keys):
        mean_f = np.mean([x['F1'][it] for x in scores])
        print('Mean {}: {}'.format(key, mean_f))


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024 * 4, rlimit[1]))
    main()
