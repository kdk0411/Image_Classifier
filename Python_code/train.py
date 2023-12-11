import torch
import timm
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from config import load_config


def train(data_loader):

  config = load_config('../config.json')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = timm.create_model(model_name=config['model_name'], pretrained=True, num_classes=7).to(device)
  class_num = [329, 205, 235, 134, 151, 245, 399]
  class_weight = torch.tensor(np.max(class_num)/class_num).to(device, dtype=torch.float)
  criterion = nn.CrossEntropyLoss(weight=class_weight)

  feat_extractor = [m for n, m in model.named_parameters() if "fc" not in n]
  classifier = [p for p in model.fc.parameters()]
  params = [
    {"params": feat_extractor, "lr": config['lr']*0.5},
    {"params": classifier, "lr": config['lr']}
  ]
  optimizer = Adam(params, lr=config['lr'])

  scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

  result = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
  }

  train_loader = data_loader["train_loader"]
  val_loader = data_loader["val_loader"]

  for epoch_idx in range(1, config['epoch'] + 1):
    model.train()

    iter_train_loss = []
    iter_val_loss = []
    iter_val_acc = []
    iter_val_f1 = []

    for iter_idx, (train_imgs, train_labels) in enumerate(train_loader, 1):
      train_imgs, train_labels = train_imgs.to(device, dtype=torch.float), train_labels.to(device)

      optimizer.zero_grad()

      train_pred = model(train_imgs)
      train_loss = criterion(train_pred, train_labels)
      train_loss.backward()

      optimizer.step()
      iter_train_loss.append(train_loss.cpu().item())

      print(f"[Epoch {epoch_idx}/{config['epoch']}] model training iteration {iter_idx}/{len(train_loader)}     ",
                end="\r")

      with torch.no_grad():
        for iter_idx, (val_imgs, val_labels) in enumerate(val_loader, 1):
          model.eval()

          val_imgs, val_labels = val_imgs.to(device, dtype=torch.float), val_labels.to(device)

          val_pred = model(val_imgs)
          val_loss = criterion(val_pred, val_labels)

          iter_val_loss.append(val_loss.cpu().item())

          val_pred_c = val_pred.argmax(dim=1)
          iter_val_acc.extend((val_pred_c == val_labels).cpu().tolist())

          iter_f1_score = f1_score(y_true=val_labels.cpu().numpy(), y_pred=val_pred_c.cpu().numpy(), average="macro")
          iter_val_f1.append(iter_f1_score)

          print(
            f"[Epoch {epoch_idx}/{config['epoch']}] model validation iteration {iter_idx}/{len(val_loader)}     ",
            end="\r"
          )
      epoch_train_loss = np.mean(iter_train_loss)
      epoch_val_loss = np.mean(iter_val_loss)
      epoch_val_acc = np.mean(iter_val_acc)
      epoch_val_f1 = np.mean(iter_val_f1)

      result["train_loss"].append(epoch_train_loss)
      result["val_loss"].append(epoch_val_loss)
      result["val_acc"].append(epoch_val_acc)
      result["val_f1"].append(epoch_val_f1)

      scheduler.step()

      print(
        f"[Epoch {epoch_idx}/{config['epoch']}] "
        f"train loss : {epoch_train_loss:.4f} | "
        f"valid loss : {epoch_val_loss:.4f} | valid acc : {epoch_val_acc:.2f}% | valid f1 score : {epoch_val_f1:.4f}"
      )

      Best_Model = None
      Best_f1 = 0
      stop_count = 0

      if epoch_val_f1 > Best_f1:
        Best_f1 = epoch_val_f1
        Best_Model = model.state_dict()
        stop_count = 0
      else:
        stop_count += 1

      if stop_count == config['early_stop']:
        print("early stoped." + " " * 30)
        break

    return result, Best_Model