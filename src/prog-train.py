# link to VirtualHome
import sys
sys.path.append('../../virtualhome/simulation')

import time
import resource
import numpy as np
from termcolor import colored
from multiprocessing import Pool

import torch

from helper import Constant, LinearStep, LCS
from program.utils import setup, log, summary, save, save_dict
from program.dataset import get_prog_dataset
from program.dataset import prog_collate_fn as collate_fn
from program.dataset import prog_to_cuda_fn as to_cuda_fn
from network.encoder_decoder import Sketch2Program


def run_one_iteration(model, optim, batch_data,
                      train_args, args, dset, pool):

    model.train()
    # update `p` in schedule sampling
    train_args['p'].update()

    optim.zero_grad()
    loss, _, _, info = model(
        batch_data, inference=False, sampling='argmax', **train_args)
    loss.backward()

    optim.step()

    return batch_data, info


def train(
        args,
        model,
        optim,
        train_loader,
        get_next_data_fn,
        checkpoint_dir,
        writer):

    # Train
    print(colored('Start training...', 'red'))
    pool = Pool(processes=max(args.n_workers, 1))

    train_args = {}
    if args.training_scheme == 'schedule_sampling':
        p = LinearStep(args.schedule_sampling_p_max,
                       args.schedule_sampling_p_min,
                       args.schedule_sampling_steps)
    else:
        p = Constant(1.)
    train_args.update({'p': p})
    dset = train_loader.dataset

    def _train_loop():
        iter = 0
        log_t1 = time.time()
        summary_t1 = time.time()

        while iter <= args.train_iters:

            for batch_data in train_loader:
                
                results = run_one_iteration(
                    model, optim, batch_data, train_args, args, dset, pool)
                batch_data, info = results

                # log
                if iter % 100 == 0:
                    log(args, iter, log_t1, info)
                    if iter > 0:
                        log_t1 = time.time()
                
                # summary (train set)
                if iter % 100 == 0:
                    fps = 100. / (time.time() - summary_t1)
                    summary(
                        args,
                        writer,
                        info,
                        train_args,
                        model,
                        batch_data,
                        train_loader.dataset,
                        pool,
                        'train',
                        fps=fps)
                    if iter > 0:
                        summary_t1 = time.time()

                # summary (test set)
                if iter % 100 == 0:
                    batch_data = get_next_data_fn()
                    summary(
                        args,
                        writer,
                        None,
                        None,
                        model,
                        batch_data,
                        train_loader.dataset,
                        pool,
                        'test')
                
                # save
                if iter % 500 == 0:
                    save(args, iter, checkpoint_dir, model)
                
                iter += 1
                del info

    
    _train_loop()
    pool.close()
    pool.join()


def main():

    # setup the config variables
    args, checkpoint_dir, writer, model_config = setup(train=True)
    
    # get dataset
    train_dset, test_dset = get_prog_dataset(args, train=True)
    save_dict(checkpoint_dir, train_dset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True)

    # loader for the testing set
    def _loader():
        while True:
            for batch_data in test_loader:
                yield batch_data
    get_next_data_fn = _loader().__iter__().__next__

    # initialize model and optim
    model = Sketch2Program(train_dset, **model_config)
    optim = torch.optim.Adam(model.parameters(), args.model_lr_rate)
    if args.gpu_id is not None:
        model.cuda()
        model.set_to_cuda_fn(to_cuda_fn)

    # main loop
    if args.checkpoint is not None:
        model.load(args.checkpoint, verbose=True)

    train(
        args,
        model,
        optim,
        train_loader,
        get_next_data_fn,
        checkpoint_dir,
        writer)


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
