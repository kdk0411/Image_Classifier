import torch
from tqdm import tqdm

def Model_Train(model, data_loader, loss_fn, optimizer, device):
  model.train()

  running_loss = 0
  corr = 0

  progress_bar = tqdm(data_loader, desc='Training', leave=False)

  for img, label in progress_bar:
    img, label = img.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(img)
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
    _, pred = output.max(dim=1)

    corr += pred.eq(label).sum().item()

    running_loss += loss.item() * img.size(0)
  acc = corr / len(data_loader.dataset)

  return running_loss / len(data_loader.dataset), acc

def Model_Evaluate(model, data_loader, loss_fn, device):
  model.eval()
  with torch.no_grad():
    corr = 0
    running_loss = 0
    for img, label in data_loader:
      img, label = img.to(device), label.to(device)
      output = model(img)
      _, pred = output.max(dim=1)
      corr += torch.sum(pred.eq(label)).item()
      running_loss += loss_fn(output, label).item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc