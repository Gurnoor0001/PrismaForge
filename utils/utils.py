from torch.utils.data import Dataset
import os 
from PIL import Image
import torch
import torchvision.transforms as transform


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform= None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith((".jpg", ".jpeg", ".png"))]
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(img)

        return image


def get_transform(img_size, crop = False, final_size = 512):
    transform_list = []

    if img_size > 0:
        transform_list.append(transform.Resize(img_size))

    if crop:
        transform_list.append(transform.RandomCrop(img_size))
    else:
        transform_list.append(transform.RandomCrop(final_size))

    transform_list.append(transform.ToTensor())

    return transform.Compose(transform_list)

    
        
def adaptive_instance_normalization(content_feats, style_feats):
    
    # [batch_size, channels, h, w]
    size = content_feats.size()
    style_mean, style_std = compute_mean_std(style_feats)
    content_mean, content_std = compute_mean_std(content_feats)

    normalized_content_feats = (content_feats -content_mean.expand(size)) / content_std.expand(size) 

    return normalized_content_feats*style_std.expand(size) + style_mean.expand(size)

def compute_mean_std(feats, eps = 1e-5):
    # [batch_size, channels, h, w]
    
    size = feats.size()
    assert (len(size) == 4)

    batch_size, channels = size[:2]

    mean = torch.mean(feats, dim=[2,3], keepdim = True)
    std = torch.std(feats, dim=[2,3], keepdim = True)

    return mean, std + eps
  
