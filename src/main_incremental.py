import copy
import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce

import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var

from torch.nn.parallel import DistributedDataParallel as DDP

def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--ablation-name', type=str, default='',
                        help='ablation study results')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')

    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    parser.add_argument('--validation', default=-1.0, type=float,
                        help='portion of validation set, turn off by setting to negative')

    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    parser.add_argument('--ddp', action='store_true',
                        help='distributed data parallel')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--syncbn', action='store_true',
                        help='whether to synchronize batch norm')

    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--decay-mile-stone', nargs='+', type=int,
                        help='mile stone of learning rate decay')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='ratio of learning rate decay')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # resume parameters
    parser.add_argument('--resume-task', type=int, default=-1,
                        help='resume from which task, default (-1) means train from scratch')
    parser.add_argument('--resume-path', type=str, default='',
                        help='the path to resume from')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, decay_mile_stone=args.decay_mile_stone,
                       lr_decay=args.lr_decay, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train,
                       ddp=args.ddp, local_rank=args.local_rank)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    if args.local_rank == 0:
        print('=' * 108)
        print('Arguments =')
        for arg in np.sort(list(vars(args).keys())):
            print('\t' + arg + ':', getattr(args, arg))
        print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    ####################################################################################################################

    # Args -- Network
    from networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    if args.local_rank == 0:
        print('Approach arguments =')
        for arg in np.sort(list(vars(appr_args).keys())):
            print('\t' + arg + ':', getattr(appr_args, arg))
        print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        if args.local_rank == 0:
            print('Exemplars dataset arguments =')
            for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
                print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
            print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    if not args.ddp:
        net.cuda()
    else:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
        if args.syncbn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net.to(args.local_rank)
        net = DDP(net, device_ids=[args.local_rank])

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory, validation=args.validation,
                                                              ddp=args.ddp)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    if args.validation < 0.0:
        val_loader = tst_loader

    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)
    utils.seed_everything(seed=args.seed)

    # resume the model
    # currently don't consider ddp
    if args.resume_task != -1:
        if args.resume_task > 0:
            raise NotImplementedError("currently only implemented loading initial task")
        state_dict = torch.load(args.resume_path, map_location='cuda')
        net.add_head(taskcla[0][1])
        appr.pre_train_process(0, trn_loader[0])
        net.load_state_dict(state_dict)
        appr.post_train_process(0, trn_loader[0], val_loader[0])

    # Loop tasks
    if args.local_rank == 0:
        print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        if args.local_rank == 0:
            print('*' * 108)
            print('Task {:2d}'.format(t))
            print('*' * 108)

        if t > args.resume_task:
            # Add head for current task
            if args.ddp:
                appr.model.module.add_head(taskcla[t][1])

                appr.model = copy.deepcopy(appr.model.module)
                appr.model.to(args.local_rank)

                appr.model = DDP(appr.model, device_ids=[args.local_rank])
            else:
                net.add_head(taskcla[t][1])
                net.to(device)

            # Train
            appr.train(t, trn_loader[t], val_loader[t])
        else:
            print('skipping task: ', t)

        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            if args.local_rank == 0:
                print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
        # Save
        if args.local_rank == 0:
            print('Save at ' + os.path.join(args.results_path, full_exp_name))
            if args.ddp:
                logger.save_model(appr.model.module.state_dict(), task=t)
            else:
                logger.save_model(appr.model.state_dict(), task=t)
        if args.ddp:
            torch.distributed.barrier()

    if args.local_rank == 0:
        # Print Summary
        utils.print_summary(taskcla, acc_taw, acc_tag, forg_taw, forg_tag)
        print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
        print('Done!')

        # print arguments again
        # -------------------------------------------------------
        print('Approach arguments =')
        for arg in np.sort(list(vars(appr_args).keys())):
            print('\t' + arg + ':', getattr(appr_args, arg))
        print('=' * 108)
        # -------------------------------------------------------

        # save summary results
        # ablation_dir = args.approach + '_ablations'
        # if not os.path.isdir(ablation_dir):
        #     os.mkdir(ablation_dir)
        # utils.save_summary(os.path.join(ablation_dir, args.ablation_name), taskcla, acc_taw, acc_tag, forg_taw, forg_tag, appr_args)

        # print confusion matrix
        # -------------------------------------------------------
        print('Confusion Matrix: ')
        if args.ddp:
            num_classes = sum([head.out_features for head in appr.model.module.heads])
        else:
            num_classes = sum([head.out_features for head in appr.model.heads])
        cm = utils.compute_confusion_matrix(appr.model, val_loader, num_classes)
        with open(os.path.join(args.results_path, full_exp_name) + '/cm.npy', 'wb') as f:
            np.save(f, cm)
        # -------------------------------------------------------

    ####################################################################################################################


if __name__ == '__main__':
    main()
