#dataset.py
import torch.utils.data as data
import PIL.Image as Image
import os


def make_dataset(rootdata,roottarget):#获取img和mask的地址
    imgs = []
    filename_data = [x for x in os.listdir(rootdata)]
    for name in filename_data:
        img = os.path.join(rootdata, name)
        mask = os.path.join(roottarget, name)
        imgs.append((img, mask))#作为元组返回
    return imgs


class MyDataset(data.Dataset):
    def __init__(self, rootdata, roottarget, transform=None, target_transform=None):
        imgs = make_dataset(rootdata,roottarget)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L')#读取并转换为二值图像
        img_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)