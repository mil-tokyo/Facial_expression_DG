import os
import os.path
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from copy import deepcopy
from dataloader.Dataset import DG_Dataset
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


def random_split_dataloader(dataset_list_dir, source_domain, target_domain, batch_size, get_domain_label=False,
                            get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8, num_class=7,
                            data_server='umihebi'):
    # if data=='VLCS':
    #     split_rate = 0.7
    # else:
    #     split_rate = 0.9

    with open(os.path.join(dataset_list_dir, 'dataset_list.json')) as f:
        dataset_list = json.load(f)

    source_data_train = {'AffectNet': dataset_list['AffectNet']['train'], 'FER2013': dataset_list['FER2013']['train'],
                         'KDEF': dataset_list['KDEF']['train']}
    source_data_val = {'AffectNet': dataset_list['AffectNet']['val'], 'FER2013': dataset_list['FER2013']['val'],
                       'KDEF': dataset_list['KDEF']['val']}
    source_train = DG_Dataset(source_data=source_data_train, domain=source_domain, split='train',
                              get_domain_label=get_domain_label, get_cluster=get_cluster,
                              color_jitter=False, min_scale=min_scale, num_class=num_class, data_server=data_server)
    source_val = DG_Dataset(source_data=source_data_val, domain=source_domain, split='val',
                            get_domain_label=False, get_cluster=False,
                            color_jitter=False, min_scale=min_scale, num_class=num_class, data_server=data_server)

    # source_train, source_val = random_split(source, [int(len(source)*split_rate),
    #                                                  len(source)-int(len(source)*split_rate)])
    # source_train = deepcopy(source_train)
    # source_train.dataset.split='train'
    # source_train.dataset.set_transform('train')
    # source_train.dataset.get_domain_label=get_domain_label
    # source_train.dataset.get_cluster=get_cluster

    target_data = {'CK': dataset_list['CK']['train'] + dataset_list['CK']['val'],
                   'JAFFE': dataset_list['JAFFE']['train'] + dataset_list['JAFFE']['val']}

    target_test = DG_Dataset(source_data=target_data, domain=target_domain, split='test',
                             get_domain_label=False, get_cluster=False, num_class=num_class, data_server=data_server)

    print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))

    source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_val = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_train, source_val, target_test
