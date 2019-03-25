import argparse
import os
import re
import time
import random
import json
import copy
import numpy as np
from multiprocessing import Pool

import torch
from tensorboardX import SummaryWriter

from helper import writer_helper, average_over_list, LCS, to_cpu


patt_number = r'\((.+?)\)'
patt_object = r'\<(.+?)\>'


def grab_args():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--sketch_path', type=str, default=None,
                        help='use the ground truth sketch or predicted sketch')
    parser.add_argument('--results_prefix', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=4)

    # dataset
    parser.add_argument('--max_words', type=int, default=35)
    parser.add_argument('--use_constraint_mask', type=str2bool, default=False)

    # graph
    parser.add_argument('--graph_evolve', type=str2bool, default=True)
    parser.add_argument('--graph_hidden', type=int, default=64)
    parser.add_argument('--n_timesteps', type=int, default=2)

    # model config
    parser.add_argument(
        '--model_type', 
        type=str, 
        default='residual', 
        choices=['fc', 'gru', 'residual'])
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--sketch_hidden', type=int, default=256)
    parser.add_argument('--program_hidden', type=int, default=256)

    # train config
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument(
        '--gpu_id',
        metavar='N',
        type=str,
        nargs='+',
        help='specify the gpu id')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_iters', type=int, default=4e4)
    parser.add_argument('--model_lr_rate', type=float, default=3e-4)
    parser.add_argument('--training_scheme', type=str,
                        choices=['teacher_forcing', 'schedule_sampling'],
                        default='schedule_sampling')

    # schedule sampleing
    parser.add_argument('--schedule_sampling_p_min', type=float, default=0.3)
    parser.add_argument('--schedule_sampling_p_max', type=float, default=1.0)
    parser.add_argument('--schedule_sampling_steps', type=int, default=2e4)

    args = parser.parse_args()
    return args


def setup(train):

    def _basic_setting(args):

        # set seed
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.gpu_id is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            args.__dict__.update({'cuda': False})
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(args.gpu_id)
            args.__dict__.update({'cuda': True})
            torch.cuda.manual_seed_all(args.seed)

        if args.debug:
            args.verbose = True

    def _basic_checking(args):
        if train:
            assert args.sketch_path is None
        
    def _create_checkpoint_dir(args):

        # setup checkpoint_dir
        if args.debug:
            checkpoint_dir = 'debug'
        elif train:
            checkpoint_dir = 'checkpoint_dir'
        else:
            checkpoint_dir = 'testing_dir'

        checkpoint_dir = os.path.join(checkpoint_dir, 'sketch2program')

        args_dict = args.__dict__
        keys = sorted(args_dict)
        prefix = ['{}-{}'.format(k, args_dict[k]) for k in keys]
        prefix.remove('debug-{}'.format(args.debug))
        prefix.remove('checkpoint-{}'.format(args.checkpoint))
        prefix.remove('sketch_path-{}'.format(args.sketch_path))
        prefix.remove('gpu_id-{}'.format(args.gpu_id))
        prefix += ['use_sketch', ]
        
        checkpoint_dir = os.path.join(checkpoint_dir, *prefix)

        checkpoint_dir += '/{}'.format(time.strftime("%Y%m%d-%H%M%S"))

        return checkpoint_dir

    def _make_dirs(checkpoint_dir, tfboard_dir):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(tfboard_dir):
            os.makedirs(tfboard_dir)

    def _print_args(args):

        args_str = ''
        with open(os.path.join(checkpoint_dir, 'args.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                s = '{}: {}'.format(k, v)
                args_str += '{}\n'.format(s)
                print(s)
                f.write(s + '\n')
        print("All the data will be saved in", checkpoint_dir)
        return args_str

    args = grab_args()

    _basic_setting(args)
    _basic_checking(args)

    checkpoint_dir = _create_checkpoint_dir(args)
    tfboard_dir = os.path.join(checkpoint_dir, 'tfboard')

    _make_dirs(checkpoint_dir, tfboard_dir)
    args_str = _print_args(args)

    writer = SummaryWriter(tfboard_dir)
    writer.add_text('args', args_str, 0)
    writer = writer_helper(writer)

    model_config = {
        "embedding_dim": args.embedding_dim,
        "sketch_hidden": args.sketch_hidden,
        "program_hidden": args.program_hidden,
        "max_words": args.max_words,
        "graph_hidden": args.graph_hidden,
        "graph_evolve": args.graph_evolve,
        "n_timesteps": args.n_timesteps,
        "model_type": args.model_type
    }

    return args, checkpoint_dir, writer, model_config


def log(args, i, log_t1, info):

    fps = 100. / (time.time() - log_t1)

    str_to_show = '[#] Iter({:5})\tTotal loss: {:.2f},\t'.format(
        i, info['total_loss'])
    str_to_show += 'action loss: {:.2f},\tobject1 loss: {:.2f},\tobject2 loss: {:.2f}'.format(
        info['action_loss'], info['object1_loss'], info['object2_loss'])

    if i > 0:
        str_to_show += '\tfps: {:.2f}'.format(fps)
        log_t1 = time.time()

    print(str_to_show)


def evalaute_executability(program_predict_str, init_graph_dicts, pool):

    from evolving_graph.check_programs import check_executability
    
    mp_inputs = [[i, j] for i, j in zip(program_predict_str, init_graph_dicts)]
    simulator_results = pool.map(check_executability, mp_inputs)
    return simulator_results


def _extract_instance_from_program(program):

    program = program.replace(' <<eos>> (<eos>)', '').replace(' <<none>> (<none>)', '')
    obj_list, id_list = [], []
    for line in program.split(', '):
        obj_match = re.search(patt_object, line)
        while obj_match:
            obj = obj_match.group(1)
            obj_list.append(obj)
            obj_match = re.search(patt_object, obj_match.string[obj_match.end(1):])

        id_match = re.search(patt_number, line)
        while id_match:
            id = id_match.group(1)
            if '.' in id:
                id_list.append(int(id.split('.')[1]))
            else:
                id_list.append(int(id))

            id_match = re.search(patt_number,
                                 id_match.string[id_match.end(1):])

    object_instance = []
    for obj, id in zip(obj_list, id_list):
        object_instance.append((obj, id))
    return list(set(object_instance))


def evaluate_f1_scores(inp):
    """
        graph1_state: the prediction graph
        graph2_state: the ground truth graph
        program1_id_list: the id that appear in prediction (exclude character node)
        program2_id_list: the id that appear in the ground truth (exclude character node)
    """

    script1, graph1_state, script2, graph2_state = inp

    if graph1_state is None:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    program1_instance_list = _extract_instance_from_program(script1)
    program2_instance_list = _extract_instance_from_program(script2)    # gt
    # find all possible final graphs
    character_id = [
        i["id"] for i in filter(
            lambda v: v['class_name'] == 'character',
            graph1_state["nodes"])][0]
    program1_instance_list += [('character', character_id)]
    program2_instance_list += [('character', character_id)]

    # trim out the room
    room_id = [
        i["id"] for i in filter(
            lambda v: v['category'] == 'Rooms',
            graph1_state["nodes"])]
    door_id = [
        i["id"] for i in filter(
            lambda v: 'door' in v['class_name'],
            graph1_state["nodes"])]

    program1_instance_list_wo_room_and_door = set([i for i in filter(
        lambda v: v[1] not in room_id and v[1] not in door_id, program1_instance_list)])
    program2_instance_list_wo_room_and_door = set([i for i in filter(
        lambda v: v[1] not in room_id and v[1] not in door_id, program2_instance_list)])

    program1_id_list_wo_room_and_door = [i[1] for i in program1_instance_list_wo_room_and_door]
    # finding interchangeable objects
    interchangeable_list = []
    for instance in program2_instance_list_wo_room_and_door:
        name, _ = instance
        object_that_is_interchangeable = []
        for node in graph2_state["nodes"]:
            if node["class_name"] == name:
                object_that_is_interchangeable.append(node["id"])
        interchangeable_list.append(object_that_is_interchangeable)

    program2_id_list_wo_room_and_door = [i[1] for i in program2_instance_list_wo_room_and_door]
    permutation_program2_id_list_wo_room_and_door = []

    for i, objs in enumerate(interchangeable_list):
        for obj in objs:
            temp = copy.copy(program2_id_list_wo_room_and_door)
            temp[i] = obj
            if obj not in permutation_program2_id_list_wo_room_and_door:
                permutation_program2_id_list_wo_room_and_door.append(temp)

    def _convert_to_tuple(graph, id_list):

        attribute_tuples = []
        for node in graph["nodes"]:
            if node["id"] in id_list:
                for state in node["states"]:
                    attribute_tuples.append((node["id"], state))

        relation_tuples = []
        for edge in graph["edges"]:
            if edge["from_id"] in id_list or edge["to_id"] in id_list:
                relation_tuples.append(
                    (edge["relation_type"], edge["from_id"], edge["to_id"]))

        return attribute_tuples, relation_tuples

    def _find_match(
            attribute_pred_tuple,
            relation_pred_tuple,
            attribute_gt_tuple,
            relation_gt_tuple):

        attribute_match = 0
        for tuple in attribute_pred_tuple:
            if tuple in attribute_gt_tuple:
                attribute_match += 1

        relation_match = 0
        for tuple in relation_pred_tuple:
            if tuple in relation_gt_tuple:
                relation_match += 1

        return attribute_match, relation_match

    def _compute_scores(
            _program1_id_list_wo_room_and_door,
            _program2_id_list_wo_room_and_door):

        attribute_pred_tuple, relation_pred_tuple = _convert_to_tuple(
            graph1_state, _program1_id_list_wo_room_and_door)
        attribute_gt_tuple, relation_gt_tuple = _convert_to_tuple(
            graph2_state, _program2_id_list_wo_room_and_door)
        attribute_pred_tuple = set(attribute_pred_tuple)
        relation_pred_tuple = set(relation_pred_tuple)
        attribute_gt_tuple = set(attribute_gt_tuple)
        relation_gt_tuple = set(relation_gt_tuple)
        attribute_match, relation_match = _find_match(
            attribute_pred_tuple, relation_pred_tuple, attribute_gt_tuple, relation_gt_tuple)

        if attribute_match != 0:
            attribute_precision = attribute_match / len(attribute_pred_tuple)
            attribute_recall = attribute_match / len(attribute_gt_tuple)
            attribute_f1 = 2 * attribute_precision * attribute_recall / \
                (attribute_precision + attribute_recall)
        else:
            attribute_precision, attribute_recall, attribute_f1 = 0, 0, 0

        if relation_match != 0:
            relation_precision = relation_match / len(relation_pred_tuple)
            relation_recall = relation_match / len(relation_gt_tuple)
            relation_f1 = 2 * relation_precision * relation_recall / \
                (relation_precision + relation_recall)
        else:
            relation_precision, relation_recall, relation_f1 = 0, 0, 0

        if (relation_match + attribute_match) != 0:
            total_precision = (relation_match + attribute_match) / \
                (len(attribute_pred_tuple) + len(relation_pred_tuple))
            total_recall = (relation_match + attribute_match) / \
                (len(attribute_gt_tuple) + len(relation_gt_tuple))
            total_f1 = 2 * total_precision * total_recall / \
                (total_precision + total_recall)
        else:
            total_precision, total_recall, total_f1 = 0, 0, 0

        return attribute_precision, relation_precision, total_precision, \
            attribute_recall, relation_recall, total_recall, \
            attribute_f1, relation_f1, total_f1

    max_scores = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    for program2_id_list_wo_room_and_door in permutation_program2_id_list_wo_room_and_door:
        scores = _compute_scores(
            program1_id_list_wo_room_and_door,
            program2_id_list_wo_room_and_door)
        if sum(scores[-3:]) > sum(max_scores[-3:]):
            max_scores = scores

    return max_scores


def decode_program_str(program_predict, dset):

    batch_action, batch_object1, batch_object2, batch_node_name_list = program_predict
    program_predict_str = dset.interpret(
        batch_action,
        batch_object1,
        batch_object2,
        batch_node_name_list,
        with_sos=False)

    return program_predict_str


def _evaluate_lcs(inp):

    pred, gt = inp

    pred = pred.replace(') <<none>> (<none>)', ')').replace('] <<none>> (<none>)', ']')
    pred = pred.replace(') <<eos>> (<eos>)', ')').replace('] <<eos>> (<eos>)', ']')
    gt = gt.replace(') <<none>> (<none>)', ')').replace('] <<none>> (<none>)', ']')
    gt = gt.replace(') <<eos>> (<eos>)', ')').replace('] <<eos>> (<eos>)', ']')

    pred = pred.split(', ')
    gt = gt.split(', ')
    lcs_score = len(LCS(pred, gt)) / (float(max(len(pred), len(gt))))

    return lcs_score


def evaluate_lcs(program_predict_str, program_str, pool):
    mp_inputs = [[i, j] for i, j in zip(program_predict_str, program_str)]
    lcs_score_list = pool.map(_evaluate_lcs, mp_inputs)
    return lcs_score_list


def get_gt_program_str(data, dset):

    batch_action, batch_object1, batch_object2 = data[0][2:]
    batch_action = to_cpu(batch_action)
    batch_object1 = to_cpu(batch_object1)
    batch_object2 = to_cpu(batch_object2)
    batch_node_name_list = data[2][1]
    program_str = dset.interpret(
        batch_action,
        batch_object1,
        batch_object2,
        batch_node_name_list,
        with_sos=True)

    return program_str


def decode_f1_results(info, f1_results):

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
    values = []
    for i in range(len(keys)):
        values.append(average_over_list([j[i] for j in f1_results]))

    for k, v in zip(keys, values):
        info.update({k: v})


def summary_inference(
        args, 
        model,
        batch_data,
        dset,
        pool):

    with torch.no_grad():

        # summary for interence info
        model.eval()
        _, _, program_predict, inference_info = model(
            batch_data, inference=True, use_constraint_mask=args.use_constraint_mask)

        program_predict_str = decode_program_str(program_predict, dset)
        program_str = get_gt_program_str(batch_data, dset)
        
        # evaluate lcs
        lcs_score_list = evaluate_lcs(program_predict_str, program_str, pool)
        lcs_score = average_over_list(lcs_score_list)
        inference_info.update({'lcs_score': lcs_score})

        # check executability
        init_graph_dicts = batch_data[2][-2]
        simulator_results = evalaute_executability(
            program_predict_str, init_graph_dicts, pool)

        parsibility = [result[0] for result in simulator_results]
        executability = [result[1] for result in simulator_results]
        parsibility = np.mean(np.array(parsibility) + 0)
        executability = np.mean(np.array(executability) + 0)

        inference_info.update({'parsibility': parsibility})
        inference_info.update({'executability': executability})

        final_states = [result[2] for result in simulator_results]
        gt_final_graph_dict = batch_data[2][-1]
        mp_inputs = [[i, j, k, l] for i, j, k, l in zip(
            program_predict_str, final_states, program_str, gt_final_graph_dict)]
        f1_results = pool.map(evaluate_f1_scores, mp_inputs)
        decode_f1_results(inference_info, f1_results)

        # summary text for properties
        _, _, batch_sketch_action, batch_sketch_object1, batch_sketch_object2, \
                    batch_sketch_index1, batch_sketch_index2 = batch_data[1]

        batch_sketch_action = to_cpu(batch_sketch_action)
        batch_sketch_object1 = to_cpu(batch_sketch_object1)
        batch_sketch_object2 = to_cpu(batch_sketch_object2)
        batch_sketch_index1 = to_cpu(batch_sketch_index1)
        batch_sketch_index2 = to_cpu(batch_sketch_index2)
        sketch_str = dset.interpret_sketch(
            batch_sketch_action,
            batch_sketch_object1,
            batch_sketch_object2,
            batch_sketch_index1,
            batch_sketch_index2,
            with_sos=True)

        sketch_str = [
            i.replace(') <<none>> (<none>)', ')').replace('] <<none>> (<none>)', ']') for i in sketch_str]
        sketch_str = [
            i.replace(') <<eos>> (<eos>)', ')').replace('] <<eos>> (<eos>)', ']') for i in sketch_str]

        node_names = program_predict[3]

    return inference_info, program_predict_str, program_str, sketch_str, f1_results, simulator_results, node_names


def program_text_helper(
        program_str,
        sketch_str,
        program_predict_str,
        info,
        verbose,
        i=0):

    # sample the first program in the batch
    str_to_show = '**Sketch({})**: {}\n\n**GT({})**: {}\n\n**PD({})**: {}\n\n'.format(
        len(sketch_str[i].split(',')), sketch_str[i],
        len(program_str[i].split(',')), program_str[i],
        len(program_predict_str[i].split(',')), program_predict_str[i])

    total_loss = info['action_loss_per_program'][i] + \
        info['object1_loss_per_program'][i] + \
        info['object2_loss_per_program'][i]

    str_to_show += 'Total loss: **{:.2f}**, '.format(total_loss)
    str_to_show += 'action loss: **{:.2f}**, object1 loss: **{:.2f}**, object2 loss: **{:.2f}**'.format(
        info['action_loss_per_program'][i], info['object1_loss_per_program'][i], info['object2_loss_per_program'][i])

    str_to_show = str_to_show.replace('>', r'\>')
    str_to_show = str_to_show.replace('<', r'\<')
    str_to_show = str_to_show.replace('_', '.')

    if verbose:
        print(str_to_show)
    return str_to_show


def summary(
        args,
        writer,
        info,
        train_args,
        model,
        batch_data,
        dset,
        pool,
        postfix,
        fps=None):

    if postfix == 'train':
        info.update({'schedule_sampling_p': train_args['p'].v})
        model.write_summary(writer, info, postfix=postfix)
    elif postfix == 'test':
        info, program_predict_str, program_str, sketch_str, _, _, _ = summary_inference(
            args, 
            model,
            batch_data,
            dset,
            pool)

        model.write_summary(writer, info, postfix=postfix)
        program_text_summary_to_show = program_text_helper(
            program_str, sketch_str, program_predict_str, info, args.verbose)

        writer.text_summary(
            'Sketch2Program-{}/output_text'.format(postfix),
            program_text_summary_to_show)
    else:
        raise ValueError

    if fps:
        writer.scalar_summary('General/fps', fps)


def save_dict(checkpoint_dir, dset):

    _dict = {
        'action_dict': dset.action_dict.word2idx,
        'object_dict': dset.object_dict.word2idx,
        'index_dict': dset.index_dict.word2idx
    }

    fp = open(os.path.join(checkpoint_dir, 'dict.json'), 'w')
    json.dump(_dict, fp)
    fp.close()


def save(args, i, checkpoint_dir, model):

    save_path = '{}/sketch2program-{}.ckpt'.format(checkpoint_dir, i)
    model.save(save_path, True)
