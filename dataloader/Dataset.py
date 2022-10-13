import torch
from torch.utils.data import Dataset
import sys
import os
from torchvision import transforms
# from torchvision.datasets.folder import make_dataset, default_loader
import numpy as np
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class DG_Dataset(Dataset):
    def __init__(self, source_data, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True,
                 min_scale=0.8, num_class=7, data_server='umihebi'):
        self.source_data = source_data
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.num_class = num_class
        self.data_server = data_server
        # self.loader = default_loader
        
        self.load_dataset()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        # image = self.loader(path)
        if self.data_server != 'umihebi':
            path = path.replace('umihebi', self.data_server)
        image = Image.open(path)
        if image.mode != 'L':
            image = image.convert('L')
        if image.size != (256, 256):
            image = image.resize((256, 256))
        image = self.transform(image)
        image = torch.cat([image, image, image], dim=0)

        if target == 'neutral':
            target = 0
        elif target == 'happy':
            target = 1
        elif target == 'anger' or target == 'angry':
            target = 2
        elif target == 'sad':
            target = 3
        elif target == 'disgust':
            target = 4
        elif target == 'fear':
            target = 5
        elif target == 'surprise':
            target = 6

        output = [image, target]
        
        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
            
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)


        return tuple(output)
    
    # def find_classes(self, dir_name):
    #     if sys.version_info >= (3, 5):
    #         # Faster and available in Python 3.5 and above
    #         classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
    #     else:
    #         classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     return classes, class_to_idx
    
    def load_dataset(self):
        total_samples = []
        self.domains = np.zeros(0)
        
        # classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        # self.num_class = len(classes)
        # for i, item in enumerate(self.domain):
        #     path = self.root_dir + item + '/'
        #     samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        #     total_samples.extend(samples)
        #     self.domains = np.append(self.domains, np.ones(len(samples)) * i)
        for i, domain in enumerate(self.source_data):
            data_num = len(self.source_data[domain])
            total_samples.extend(self.source_data[domain])
            self.domains = np.append(self.domains, np.ones(data_num) * i)
            
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list
            
    def set_transform(self, split):
        if split == 'train':
            if self.color_jitter:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(5, sigma=(0.1, 0.2)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, ], [0.5, ])
                ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, ], [0.5, ])
            ])
        else:
            raise Exception('Split must be train or val or test!!')