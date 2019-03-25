import argparse
import random
import time
import os
import json
import numpy as np


import torch
from tensorboardX import SummaryWriter

from program.utils import evaluate_lcs
from helper import to_cpu, average_over_list, writer_helper


def grab_args():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=4)

    # dataset
    parser.add_argument('--max_words', type=int, default=15)
    parser.add_argument('--src', type=str, choices=['desc', 'demo'])

    # model config
    parser.add_argument(
        '--model_type',
        type=str,
        default='max',
        choices=[
            'avg',
            'max',
            'trn'])
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--sketch_hidden', type=int, default=128)
    parser.add_argument('--demo_hidden', type=int, default=128)
    parser.add_argument('--desc_hidden', type=int, default=256)

    # train config
    parser.add_argument(
        '--gpu_id',
        metavar='N',
        type=str,
        nargs='+',
        help='specify the gpu id')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_lr_rate', type=float, default=3e-4)
    parser.add_argument('--training_scheme', type=str,
                        choices=['teacher_forcing', 'schedule_sampling'],
                        default='schedule_sampling')

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
        pass

    def _create_checkpoint_dir(args):

         # setup checkpoint_dir
        if args.debug:
            checkpoint_dir = 'debug'
        elif train:
            checkpoint_dir = 'checkpoint_dir'
        else:
            checkpoint_dir = 'testing_dir'

        if args.src == 'desc':
            checkpoint_dir = os.path.join(checkpoint_dir, 'desc2sketch')
        elif args.src == 'demo':
            checkpoint_dir = os.path.join(checkpoint_dir, 'demo2sketch')
        else:
            raise ValueError

        args_dict = args.__dict__
        keys = sorted(args_dict)
        prefix = ['{}-{}'.format(k, args_dict[k]) for k in keys]
        prefix.remove('debug-{}'.format(args.debug))
        prefix.remove('checkpoint-{}'.format(args.checkpoint))
        prefix.remove('gpu_id-{}'.format(args.gpu_id))

        if args.src == 'desc':
            prefix.remove('model_type-{}'.format(args.model_type))
            prefix.remove('demo_hidden-{}'.format(args.demo_hidden))
        elif args.src == 'demo':
            prefix.remove('desc_hidden-{}'.format(args.desc_hidden))
        else:
            raise ValueError

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
    if args.src == 'desc':
        args.__dict__.update({'train_iters': 5e4})
        args.__dict__.update({'schedule_sampling_p_min': 0.})
        args.__dict__.update({'schedule_sampling_p_max': 1.0})
        args.__dict__.update({'schedule_sampling_steps': 2e4})
    elif args.src == 'demo':
        args.__dict__.update({'train_iters': 1e4})
        args.__dict__.update({'schedule_sampling_p_min': 0.})
        args.__dict__.update({'schedule_sampling_p_max': 1.0})
        args.__dict__.update({'schedule_sampling_steps': 1e4})
    else:
        raise ValueError

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
        "model_type": args.model_type,
        "embedding_dim": args.embedding_dim,
        "sketch_hidden": args.sketch_hidden,
        "max_words": args.max_words,
    }
    if args.src == 'desc':
        model_config.update({"desc_hidden": args.desc_hidden})
    elif args.src == 'demo':
        model_config.update({"demo_hidden": args.demo_hidden})
    else:
        raise ValueError


    return args, checkpoint_dir, writer, model_config


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
        # log the training results (teacher forcing or schedule sampling)
        info.update({'schedule_sampling_p': train_args['p'].v})
        model.write_summary(writer, info, postfix=postfix)
    elif postfix == 'test':
        # log the testing results (free run)
        info, _, _, _ = summary_inference(
            model,
            batch_data,
            dset,
            pool)

        model.write_summary(writer, info, postfix=postfix)
    else:
        raise ValueError

    if fps:
        writer.scalar_summary('General/fps', fps)


def summary_inference(
        model,
        batch_data,
        dset,
        pool):

    with torch.no_grad():

        # summary for inference info
        model.eval()
        _, sketch_predict, info = model(batch_data, inference=True)
        batch_action, batch_object1, batch_object2 = sketch_predict
        sketch_predict_str = dset.interpret_sketch(
            batch_action, batch_object1, batch_object2, with_sos=False)

        batch_sketch_action, batch_sketch_object1, batch_sketch_object2, _, _ = batch_data[1][2:]
        batch_sketch_action = to_cpu(batch_sketch_action)
        batch_sketch_object1 = to_cpu(batch_sketch_object1)
        batch_sketch_object2 = to_cpu(batch_sketch_object2)
        sketch_str = dset.interpret_sketch(
            batch_sketch_action,
            batch_sketch_object1,
            batch_sketch_object2,
            with_sos=True)

        # evaluate lcs
        lcs_score_list = evaluate_lcs(sketch_predict_str, sketch_str, pool)
        lcs_score = average_over_list(lcs_score_list)
        info.update({'lcs_score': lcs_score})

        sketch_predict_str = [
            i.replace(') <<none>> (<none>)', ')').replace('] <<none>> (<none>)', ']') for i in sketch_predict_str]
        sketch_predict_str = [
            i.replace(') <<eos>> (<eos>)', ')').replace('] <<eos>> (<eos>)', ']') for i in sketch_predict_str]

        sketch_str = [
            i.replace(') <<none>> (<none>)', ')').replace('] <<none>> (<none>)', ']') for i in sketch_str]
        sketch_str = [
            i.replace(') <<eos>> (<eos>)', ')').replace('] <<eos>> (<eos>)', ']') for i in sketch_str]

    return info, sketch_predict_str, sketch_str, lcs_score_list


def save(args, i, checkpoint_dir, model):

    if args.src == 'desc':
        save_path = '{}/desc2sketch-{}.ckpt'.format(checkpoint_dir, i)
    elif args.src == 'demo':
        save_path = '{}/demo2sketch-{}.ckpt'.format(checkpoint_dir, i)
    else:
        raise ValueError
        
    model.save(save_path, True)


def save_dict(checkpoint_dir, dset):
    
    _dict = {
        'action_dict': dset.action_dict.word2idx,
        'object_dict': dset.object_dict.word2idx,
    }
    if hasattr(dset, 'word_dict'):
        _dict.update({'word_dict': dset.word_dict.word2idx})

    fp = open(os.path.join(checkpoint_dir, 'dict.json'), 'w')
    json.dump(_dict, fp)
    fp.close()

