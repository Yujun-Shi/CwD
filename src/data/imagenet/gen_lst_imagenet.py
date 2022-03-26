import os
import numpy as np

subset_num = 100

root_dir = 'train'
with open(root_dir+'_'+str(subset_num)+'.txt', 'w') as f:
    classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    seed = 1993
    np.random.seed(seed)
    subset_classes = np.random.choice(classes, subset_num, replace=False)

    for class_id, class_name in enumerate(subset_classes):
        folder_name = os.path.join(root_dir, class_name)
        for img_name in sorted(os.listdir(folder_name)):
            write_line = os.path.join(root_dir, class_name, img_name)
            write_line += ' ' + str(class_id) + '\n'
            f.write(write_line)

root_dir = 'val'
with open(root_dir+'_'+str(subset_num)+'.txt', 'w') as f:
    classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    seed = 1993
    np.random.seed(seed)
    subset_classes = np.random.choice(classes, subset_num, replace=False)

    for class_id, class_name in enumerate(subset_classes):
        folder_name = os.path.join(root_dir, class_name)
        for img_name in sorted(os.listdir(folder_name)):
            write_line = os.path.join(root_dir, class_name, img_name)
            write_line += ' ' + str(class_id) + '\n'
            f.write(write_line)
