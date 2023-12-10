import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset

class_encoder = {
  'dog': 0,
  'elephant': 1,
  'giraffe': 2,
  'guitar': 3,
  'horse': 4,
  'house': 5,
  'person': 6
}

def img_gather(img_path):
  class_list = os.listdir(img_path)

  file_lists = []
  label_lists = []

  for class_name in class_list:
    file_list = os.listdir(os.path.join(img_path, class_name))
    file_list = list(map(lambda x: "/".join([img_path] + [class_name] + [x]), file_list))
    label_list = [class_encoder[class_name]] * len(file_list)

    file_lists.extend(file_list)
    label_lists.extend(label_list)

  file_lists = np.array(file_lists)
  label_lists = np.array(label_lists)

  return file_lists, label_lists


class TrainDataset(Dataset):
  def __init__(self, file_lists, label_lists, transforms=None):
    self.file_lists = file_lists.copy()
    self.label_lists = label_lists.copy()
    self.transforms = transforms

  def __getitem__(self, idx):
    img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.transforms:
      img = self.transforms(image=img)["image"]

    img = img.transpose(2, 0, 1)

    label = self.label_lists[idx]

    img = torch.tensor(img, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)

    return img, label

  def __len__(self):
    assert len(self.file_lists) == len(self.label_lists)
    return len(self.file_lists)


class TestDataset(Dataset):
  def __init__(self, file_lists, transforms=None):
    self.file_lists = file_lists.copy()
    self.transforms = transforms

  def __getitem__(self, idx):
    img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.transforms:
      img = self.transforms(image=img)["image"]

    img = img.transpose(2, 0, 1)

    img = torch.tensor(img, dtype=torch.float)

    return img

  def __len__(self):
    return len(self.file_lists)
