import timm
import torch
import numpy as np
from seed import random_seed
import albumentations as A
from Dataset import TestDataset
from torch.utils.data import DataLoader
from config import load_config
import os
import main


# def load_model(model, model_path):
#   model.load_state_dict(torch.load(model_path))
#   return model


def main_test():
  random_seed()
  config = load_config('config.json')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  test_transforms = A.Compose([
    A.Normalize()
  ])

  test_files = os.listdir("test/test_image")
  test_files = sorted(test_files)
  test_files = list(map(lambda x: "/".join(["test/test_image", x]), test_files))

  test_dataset = TestDataset(file_lists=test_files, transforms=test_transforms)
  test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
  import pandas as pd

  answer_logits = []

  model = timm.create_model(config['model_name'], pretrained=True, num_classes=7).to(device=device)

  for fold_idx, best_model in enumerate(main.best_models, 1):
    model.load_state_dict(best_model)
    model.eval()

    fold_logits = []

    with torch.no_grad():
      for iter_idx, test_imgs in enumerate(test_loader, 1):
        test_imgs = test_imgs.to(device)

        test_pred = model(test_imgs)
        fold_logits.extend(test_pred.cpu().tolist())

        print(f"[{fold_idx} fold] inference iteration {iter_idx}/{len(test_loader)}" + " " * 10, end="\r")

    answer_logits.append(fold_logits)

  answer_logits = np.mean(answer_logits, axis=0)
  answer_value = np.argmax(answer_logits, axis=-1)

  i = 0
  while True:
    if not os.path.isfile(os.path.join("submissions", f"submission_{i}.csv")):
      submission_path = os.path.join("submissions", f"submission_{i}.csv")
      break
    i += 1

  submission = pd.read_csv("test_answer_sample_.csv", index_col=False)
  submission["answer value"] = answer_value
  submission["answer value"].to_csv(submission_path)
  print("\nAll done.")

if __name__ == "__main__":
    main_test()