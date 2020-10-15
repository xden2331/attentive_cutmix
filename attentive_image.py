from attentive_transform import AttentiveInputTransform
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch.nn as nn
import os
from PIL import Image
import numpy as np

root = './images/'
img_paths = os.listdir(root)
model = models.resnet50(pretrained=True)
temp_model = nn.Sequential(*list(model.children())[:-2])
model = temp_model.cuda()

def select_random_image(i):
    rand_index = np.random.randint(0, len(img_paths))
    while rand_index == i:
        rand_index = np.random.randint(0, len(img_paths))
    rand_img = Image.open(img_paths[rand_index])
    return rand_img.resize((224, 224))

def top_k(self, a):
    k = 6
    idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))
        
def get_attentive_regions(image):
    """
    CIFAR return top k from 8x8
    ImageNet return top k from 7x7
    """
    x = TF.to_tensor(image).unsqueeze_(0).cuda()
    output = model(x)
    last_feature_map = output[0][-1].detach().cpu().numpy()
    return top_k(last_feature_map)

def replace_attentive_regions(rand_img, image, attentive_regions):
    """
    rand_img: the img to be replaced
    image: where the 'patches' come from
    attentive_regions: an array contains the coordinates of attentive regions
    """
    np_rand_img, np_img = np.array(rand_img), np.array(image)
    for attentive_region in attentive_regions:
        replace_attentive_region(np_rand_img, np_img, attentive_region)
    return Image.fromarray(np_rand_img)

def replace_attentive_region(self, np_rand_img, np_img, attentive_region):
    x, y = attentive_region
    x1, x2, y1, y2 = self.grid_size * x, self.grid_size * (x+1), self.grid_size * y, self.grid_size * (y+1)
    region = np_img[x1:x2, y1: y2]
    np_rand_img[x1:x2, y1:y2] = region

for i, rel_path in enumerate(img_paths):
    img_path = root + rel_path
    img = Image.open(img_path).resize((224, 224))
    rand_img = select_random_image(i)
    attentive_regions = get_attentive_regions(img)
    rand_img = replace_attentive_regions(rand_img, img, attentive_regions)
    rand_img.save('./mixed_images/{}'.format(rel_path))



