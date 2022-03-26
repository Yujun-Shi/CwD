import torch
import numpy as np

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu


class Appr(Inc_Learning_Appr):
    """Class implementing the Class Incremental Learning With Dual Memory (IL2M) approach described in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf
    """

    def __init__(self, model, device, nepochs=160, lr=0.1, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None,
                 first_task_lr=0.1, first_task_bz=128):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank,
                                   logger, exemplars_dataset)
        self.init_classes_means = []
        self.current_classes_means = []
        self.models_confidence = []
        # FLAG to not do scores rectification while finetuning training
        self.ft_train = False

        self.first_task_lr = first_task_lr
        self.first_task_bz = first_task_bz
        self.first_task = True

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: IL2M needs exemplars.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--first-task-lr', default=0.1, type=float)
        parser.add_argument('--first-task-bz', default=32, type=int)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        params = model.parameters()

        if self.first_task:
            self.first_task = False
            optimizer = torch.optim.SGD(params, lr=self.first_task_lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        print(optimizer.param_groups[0]['lr'])
        return optimizer

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if model.model.__class__.__name__ == 'ResNetCifar':
                old_block = model.model.layer3[-1]
                model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNet':
                old_block = model.model.layer4[-1]
                model.model.layer4[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNetBottleneck':
                old_block = model.model.layer4[-1]
                model.model.layer4[-1] = BottleneckNoRelu(old_block.conv1, old_block.bn1,
                                                          old_block.relu, old_block.conv2, old_block.bn2,
                                                          old_block.conv3, old_block.bn3, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")

        # Changes the new head to a CosineLinear
        model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        model.to(self.device)
        # if t > 0:
            # Share sigma (Eta in paper) between all the heads
            # Yujun: according to il2m, since we'll correct this with model confidence
            # maybe we shouldn't share sigma here.
            # model.heads[-1].sigma = model.heads[-2].sigma

            # and we probably shouldn't freeze sigma here.
            # for h in model.heads[:-1]:
            #     for param in h.parameters():
            #         param.requires_grad = False
            # model.heads[-1].sigma.requires_grad = True

        # if ddp option is activated, need to re-wrap the ddp model
        if self.ddp:
            self.model = DDP(self.model.module, device_ids=[self.local_rank])

        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t == 0:
            dset = trn_loader.dataset
            trn_loader = torch.utils.data.DataLoader(dset,
                    batch_size=self.first_task_bz,
                    sampler=trn_loader.sampler,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory)

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            dset = trn_loader.dataset + self.exemplars_dataset
            if self.ddp:
                trn_sampler = torch.utils.data.DistributedSampler(dset, shuffle=True)
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         sampler=trn_sampler,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)
            else:
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         shuffle=True,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)


        # FINETUNING TRAINING -- contains the epochs loop
        self.ft_train = True
        super().train_loop(t, trn_loader, val_loader)
        self.ft_train = False

        if self.ddp:
            # need to change the trainloader to the original version without distributed sampler
            dset = trn_loader.dataset
            trn_loader = torch.utils.data.DataLoader(dset,
                batch_size=200, shuffle=False, num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform, self.ddp)

    def criterion(self, t, outputs, targets):
        if self.ddp:
            model = self.model.module
        else:
            model = self.model
        
        if type(outputs[0]) == dict:
            outputs = [o['wsigma'] for o in outputs]
        
        """Returns the loss value"""
        if len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - model.task_offset[t])
