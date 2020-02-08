import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
         
        if self.mode == 'train':
            self.files_cartoon = sorted(glob.glob(os.path.join(root, '%s/Cartoon' % mode) + '/*.*'))
            self.files_Blur = sorted(glob.glob(os.path.join(root, '%s/Cartoon_blur' % mode) + '/*.*'))
        else:
            self.files_cartoon = sorted(glob.glob(os.path.join(root, '%s/Photo' % mode) + '/*.*'))
            self.files_Blur = sorted(glob.glob(os.path.join(root, '%s/Photo' % mode) + '/*.*'))
        self.files_photo = sorted(glob.glob(os.path.join(root, '%s/Photo' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_photo = self.transform(Image.open(self.files_photo[index % len(self.files_photo)]))

        if self.unaligned:
            item_Blur = self.transform(Image.open(self.files_Blur[random.randint(0, len(self.files_Blur) - 1)]))
        else:
            item_Blur = self.transform(Image.open(self.files_Blur[index % len(self.files_Blur)]))

        if self.unaligned:
            item_cartoon = self.transform(Image.open(self.files_cartoon[random.randint(0, len(self.files_cartoon) - 1)]))
        else:
            item_cartoon = self.transform(Image.open(self.files_cartoon[index % len(self.files_cartoon)]))

        return {'input_photo': item_photo, 'input_cartoon_blur': item_Blur, 'input_cartoon': item_cartoon}

    def __len__(self):
        return max(len(self.files_photo), len(self.files_Blur), len(self.files_cartoon))