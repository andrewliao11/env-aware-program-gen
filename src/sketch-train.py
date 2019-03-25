import resource
import time
from termcolor import colored
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader

from helper import Constant, LinearStep
from sketch.utils import save, setup, save_dict
from sketch.utils import summary


def run_one_iteration(model, optim, batch_data, train_args, args):

    model.train()
    # update `p` in schedule sampling
    train_args['p'].update()

    optim.zero_grad()
    loss, _, info = model(batch_data, inference=False, **train_args)
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

    def _train_loop():

        iter = 0
        summary_t1 = time.time()
        while iter <= args.train_iters:
            for batch_data in train_loader:

                results = run_one_iteration(
                    model, optim, batch_data, train_args, args)
                batch_data, info = results

                # summary
                if iter % 10 == 0:
                    fps = 10. / (time.time() - summary_t1)
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

                # use random select to do evaluation now
                if iter % 50 == 0:
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

                if iter % 1000 == 0:
                    save(args, iter, checkpoint_dir, model)

                iter += 1
    
    _train_loop()    
    pool.close()
    pool.join()


def main():

    args, checkpoint_dir, writer, model_config = setup(train=True)
    
    if args.src == 'desc':
        from sketch.desc_dataset import get_dataset
        from sketch.desc_dataset import collate_fn
        from sketch.desc_dataset import to_cuda_fn
    elif args.src == 'demo':
        from sketch.demo_dataset import get_dataset
        from sketch.demo_dataset import collate_fn
        from sketch.demo_dataset import to_cuda_fn
    else:
        raise ValueError


    train_dset, test_dset = get_dataset(args, train=True)
    save_dict(checkpoint_dir, train_dset)

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        drop_last=True)

    test_loader = DataLoader(
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

    # initialize model
    if args.src == 'desc':
        from network.encoder_decoder import Desc2Sketch
        model = Desc2Sketch(train_dset, **model_config)
    elif args.src == 'demo':
        from network.encoder_decoder import Demo2Sketch
        model = Demo2Sketch(train_dset, **model_config)
    else:
        raise ValueError


    optim = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        args.model_lr_rate)
    if args.gpu_id is not None:
        model.cuda()
        model.set_to_cuda_fn(to_cuda_fn)

    # main loop
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
    main()
