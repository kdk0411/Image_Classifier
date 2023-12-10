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

    # Set data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_path = 'train'
    full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    model = CNN_Block()
    model.to(device)

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    epoch = config['epoch']
    min_loss = np.inf

    for epoch in range(epoch):
        train_loss, train_acc = Model_Train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = Model_Evaluate(model, val_loader, loss_fn, device)

        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model Success')
            min_loss = val_loss
            torch.save(model.state_dict(), 'CNN_Block.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

if __name__ == "__main__":
    main()