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

    # assume the trn_loader using naive sampler instead of distributed sampler
    def il2m(self, t, trn_loader):
        """Compute and store statistics for score rectification"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        old_classes_number = sum(model.task_cls[:t])
        classes_counts = [0 for _ in range(sum(model.task_cls))]
        models_counts = 0

        # to store statistics for the classes as learned in the current incremental state
        self.current_classes_means = [0 for _ in range(old_classes_number)]
        # to store statistics for past classes as learned in their initial states
        for cls in range(old_classes_number, old_classes_number + model.task_cls[t]):
            self.init_classes_means.append(0)
        # to store statistics for model confidence in different states (i.e. avg top-1 pred scores)
        self.models_confidence.append(0)

        # compute the mean prediction scores that will be used to rectify scores in subsequent tasks
        with torch.no_grad():
            self.model.eval()
            for images, targets in trn_loader:
                outputs = self.model(images.to(self.device))
                scores = np.array(torch.cat(outputs, dim=1).data.cpu().numpy(), dtype=np.float)
                for m in range(len(targets)):
                    if targets[m] < old_classes_number:
                        # computation of class means for past classes of the current state.
                        self.current_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                    else:
                        # compute the mean prediction scores for the new classes of the current state
                        self.init_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                        # compute the mean top scores for the new classes of the current state
                        self.models_confidence[t] += np.max(scores[m, ])
                        models_counts += 1
        # Normalize by corresponding number of images
        for cls in range(old_classes_number):
            self.current_classes_means[cls] /= classes_counts[cls]
        for cls in range(old_classes_number, old_classes_number + model.task_cls[t]):
            self.init_classes_means[cls] /= classes_counts[cls]
        self.models_confidence[t] /= models_counts

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

        # IL2M outputs rectification
        self.il2m(t, trn_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform, self.ddp)

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        if self.ft_train:
            # no score rectification while training
            hits_taw, hits_tag = super().calculate_metrics(outputs, targets)
        else:
            if self.ddp:
                model = self.model.module
            else:
                model = self.model
            # Task-Aware Multi-Head
            pred = torch.zeros_like(targets.to(self.device))
            for m in range(len(pred)):
                this_task = (model.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + model.task_offset[this_task]
            hits_taw = (pred == targets.to(self.device)).float()
            # Task-Agnostic Multi-Head
            if self.multi_softmax:
                outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            # Eq. 1: rectify predicted scores
            old_classes_number = sum(model.task_cls[:-1])
            for m in range(len(targets)):
                rectified_outputs = torch.cat(outputs, dim=1)
                pred[m] = rectified_outputs[m].argmax()
                if old_classes_number:
                    # if the top-1 class predicted by the network is a new one, rectify the score
                    if int(pred[m]) >= old_classes_number:
                        for o in range(old_classes_number):
                            o_task = int((model.task_cls.cumsum(0) <= o).sum())
                            rectified_outputs[m, o] *= (self.init_classes_means[o] / self.current_classes_means[o]) * \
                                                       (self.models_confidence[-1] / self.models_confidence[o_task])
                        pred[m] = rectified_outputs[m].argmax()
                    # otherwise, rectification is not done because an old class is directly predicted
            hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

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
