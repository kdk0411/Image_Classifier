import torch
from Train_Eval import Model_Evaluate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from Model import CNN_Block, load_config
import torch.nn as nn


def main():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  config = load_config('config.json')

  test_path = 'test'
  batch_size = config["batch_size"]
  num_workers = config["num_workers"]

  transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  test_dataset = datasets.ImageFolder(
      root=test_path,
      transform=transform)

  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers)

  model = CNN_Block()
  model.to(device)

  loss_fn = nn.CrossEntropyLoss()

  model.load_state_dict(torch.load('CNN_BLOCK_best.pth'))
  test_loss, test_acc = Model_Evaluate(model, test_loader, loss_fn, device)
  print(f'evaluation loss: {test_loss:.5f}, evaluation accuracy: {test_acc:.5f}')

if __name__ == "__main__":
    main()