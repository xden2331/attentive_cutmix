import torchvision.models as models

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image

class AttentiveTargetTransform(object):
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def __call__(self, target):
        target = [target, self.placeholder['rand_target']]
        return torch.tensor(target)

class AttentiveInputTransform(object):
    def __init__(self, dataname, dataset, placeholder, k = 6):
        model = models.resnet50(pretrained=True)
        self.dataname = dataname
        self.dataset = dataset
        self.placeholder = placeholder
        self.k = k
        if dataname == 'imagenet':
            self.grid_size = 32
            temp_model = nn.Sequential(*list(model.children())[:-2])
        else:
            self.grid_size = 4
            temp_model = nn.Sequential(*list(model.children())[:-5])
        self.model = temp_model
    
    def __call__(self, image):
        rand_index = np.random.randint(0, len(self.dataset))
        rand_img, rand_target = self.dataset[rand_index]
        self.placeholder['rand_target'] = rand_target
        attentive_regions = self._get_attentive_regions(image)
        rand_img = self._replace_attentive_regions(rand_img, image, attentive_regions)
        return rand_img
    
    def _replace_attentive_regions(self, rand_img, image, attentive_regions):
        """
        rand_img: the img to be replaced
        image: where the 'patches' come from
        attentive_regions: an array contains the coordinates of attentive regions
        """
        np_rand_img, np_img = np.array(rand_img), np.array(image)
        for attentive_region in attentive_regions:
            self._replace_attentive_region(np_rand_img, np_img, attentive_region)
        return Image.fromarray(np_rand_img)

    def _replace_attentive_region(self, np_rand_img, np_img, attentive_region):
        x, y = attentive_region
        x1, x2, y1, y2 = self.grid_size * x, self.grid_size * (x+1), self.grid_size * y, self.grid_size * (y+1)
        region = np_img[x1:x2, y1: y2]
        np_rand_img[x1:x2, y1:y2] = region

    def _top_k(self, a):
        k = self.k
        idx = np.argpartition(a.ravel(),a.size-k)[-k:]
        return np.column_stack(np.unravel_index(idx, a.shape))
        
    def _get_attentive_regions(self, image):
        """
        CIFAR return top k from 8x8
        ImageNet return top k from 7x7
        """
        x = TF.to_tensor(image).unsqueeze_(0)
        output = self.model(x)
        last_feature_map = output[0][-1].detach().cpu().numpy()
        return self._top_k(last_feature_map)