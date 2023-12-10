import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import CNN_Block, CNN, DNN, load_config
from Train_Eval import Model_Train, Model_Evaluate

def main():
    config = load_config('config.json')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = 'train'
    test_path = 'test'

    full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = CNN_Block()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    epoch = config['epoch']
    max_acc = np.inf

    for epoch in range(epoch):
        train_loss, train_acc = Model_Train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = Model_Evaluate(model, test_loader, loss_fn, device)

        if max_acc < val_acc:
            print(f'[INFO] val_loss has been improved from {max_acc:.5f} to {val_acc:.5f}. Saving Model Success')
            max_acc = val_acc
            torch.save(model.state_dict(), 'CNN_BLOCK.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

if __name__ == "__main__":
    main()