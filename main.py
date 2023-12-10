import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from Model import CNN
from Train_Eval import Model_Train, Model_Evaluate


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('config.json')

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_path = '/content/drive/MyDrive/train'
test_path = '/content/drive/MyDrive/test'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform)

test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

batch_size = config["batch_size"]
num_workers = config["num_workers"]
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)

model = CNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config['lr'])
loss_fn = nn.CrossEntropyLoss()

epoch = config['epoch']
min_loss = np.inf

for epoch in range(epoch):
    train_loss, train_acc = Model_Train(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = Model_Evaluate(model, val_loader, loss_fn, device)

    if val_loss < min_loss:
        print(f'[INFO] val_loss has been imporved from {min_loss:.5f} to {val_loss:.5f}. Saving Model Success')
        min_loss = val_loss
        torch.save(model.state_dict(), 'DNN.pth')

    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')


model.load_state_dict(torch.load('DNN.pth'))
test_loss, test_acc = Model_Evaluate(model, test_loader, loss_fn, device)
print(f'evaluation loss: {test_loss:.5f}, evaluation accuracy: {test_acc:.5f}')