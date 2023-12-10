import torch
import numpy as np
import albumentations as A
from train import train
from Dataset import img_gather, TrainDataset, TestDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from config import load_config
import random


def main():
  random_seed = 42

  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)

  config = load_config('config.json')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  train_transforms = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.Normalize()
  ])

  valid_transforms = A.Compose([
    A.Normalize()
  ])
  data_lists, data_labels = img_gather("train")

  best_models = []
  k_fold_num = config['k_fold_num']

  if k_fold_num == -1:
    train_lists, valid_lists, train_labels, valid_labels = train_test_split(data_lists, data_labels, train_size=0.8,
                                                                            shuffle=True, random_state=random_seed,
                                                                            stratify=data_labels)

    train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
    valid_dataset = TrainDataset(file_lists=valid_lists, label_lists=valid_labels, transforms=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)

    data_loader = {"train_loader": train_loader, "valid_loader": valid_loader}

    print("No fold training starts ... ")
    train_result, best_model = train(data_loader)

    best_models.append(best_model)

  else:
    skf = StratifiedKFold(n_splits=k_fold_num, random_state=random_seed, shuffle=True)

    print(f"{k_fold_num} fold training starts ... ")
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(data_lists, data_labels), 1):
      print(f"- {fold_idx} fold -")
      train_lists, train_labels = data_lists[train_idx], data_labels[train_idx]
      val_lists, valid_labels = data_lists[valid_idx], data_labels[valid_idx]

      train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
      val_dataset = TrainDataset(file_lists=val_lists, label_lists=valid_labels, transforms=valid_transforms)

      train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

      data_loader = {"train_loader": train_loader, "val_loader": val_loader}

      train_result, best_model = train(data_loader)

      best_models.append(best_model)

      model_save_path = f"best_model_fold_{fold_idx}.pt"
      torch.save(best_model.state_dict(), model_save_path)
      print(f"Saved model for fold {fold_idx} to {model_save_path}")
if __name__ == "__main__":
    main()