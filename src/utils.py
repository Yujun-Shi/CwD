import os
import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(taskcla, acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    tag_acc = []
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')

            # calculate average
            task_weight = np.array([ncla for _,ncla in taskcla[0:i+1]])
            task_weight = task_weight / task_weight.sum()

            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                avg_metric = 100 * (metric[i, :i + 1]*task_weight).sum()
                print('\tAvg.:{:5.1f}% '.format(avg_metric), end='')
                if name == 'TAg Acc':
                    tag_acc.append(avg_metric)
            print()
    print('*' * 108)
    avg_tag_acc = np.array(tag_acc).mean()
    print('Average Incremental Accuracy: ', avg_tag_acc)
    print('done')

# save results of abalation study
def save_summary(save_path, taskcla, acc_taw, acc_tag, forg_taw, forg_tag, appr_args):
    """save summary of results"""
    with open(save_path, 'w') as f:
        for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
            f.write('*' * 108 + '\n')
            f.write(name + '\n')
            for i in range(metric.shape[0]):
                f.write('\t')
                for j in range(metric.shape[1]):
                    f.write('{:5.1f}% '.format(100 * metric[i, j]))
                
                # calculate average
                task_weight = np.array([ncla for _,ncla in taskcla[0:i+1]])
                task_weight = task_weight / task_weight.sum()

                if np.trace(metric) == 0.0:
                    if i > 0:
                        f.write('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()))
                else:
                    f.write('\tAvg.:{:5.1f}% '.format(100 * (metric[i, :i + 1]*task_weight).sum()))
                f.write('\n')

        # --------------- approach arguments ------------------
        f.write('*' * 108 + '\n')
        f.write('Approach arguments =\n')
        for arg in np.sort(list(vars(appr_args).keys())):
            f.write('\t' + arg + ': ' + str(getattr(appr_args, arg)) + '\n')
        f.write('=' * 108 + '\n')
        # -----------------------------------------------------

# val_loaders: a list of data loaders
def compute_confusion_matrix(model, val_loaders, num_classes):
    with torch.no_grad():        
        model.eval()
        num_classes = sum([head.out_features for head in model.heads])
        cm = np.zeros((num_classes, num_classes))
        for loader in val_loaders:
            for images, targets in loader:
                images = images.cuda()
                outputs = model(images)
                outputs = torch.cat(outputs, dim=1)

                pred = outputs.argmax(dim=1).cpu().numpy()
                targets = targets.cpu().numpy()
                cm += confusion_matrix(targets, pred, labels=np.arange(num_classes))
        return cm
