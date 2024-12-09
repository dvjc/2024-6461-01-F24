def display_setup():
    print('\n|----------------------------setup-----------------------------|')
    #print('    installed numpy (pip install numpy --user)')
    #print('    installed h5py (pip install h5py --user)')
    print('    installed pytorch (pip install torch --user)')
    print('    installed torchvision (pip install torchvision --user)')
    print('    installed skikit-learn (pip install -U scikit-learn)')
    print('    installed wandb (pip install wandb --user)')
    print('    downgraded numpy for wandb (pip install "numpy<2")')
    print('    logged into wandb (wandb login; note - need API key to use)')
    print('\n|----------------------------start-----------------------------|')
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import CNNModel
#import wandb
import datetime
import time

# Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    print("    Training:")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #wandb.log({"epoch": epoch + 1, "loss": running_loss / len(train_loader)})
        spacer = ' ' if epoch < 9 else ''
        print(f"    Epoch [{spacer}{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation Function with Precision, Recall, F1 Score for each class
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0.0)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)

    print("\n    Results:")
    print(f"cross-accuracy: {accuracy}")
    print("    Class          Precision    Recall       F1")
    #classes = ['pedestrians', 'cars', 'buses', 'traffic_signs', 'sunny', 'cloudy', 'rainy', 'snowy']
    classes = ['traffic_signs', 'snowy']
    for i, cls in enumerate(classes):
        class_spacer = ' ' * (13 - len(cls))
        print(f'    {cls}: {class_spacer}{precision[i]:.4f}       {recall[i]:.4f}       {f1[i]:.4f}')

def main():
    
    #Initialize Weights and Biases
    #wandb.init(project="autonomous-driving-object-detection")
    
    # Data Transformation and Loading
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(root='data/training', transform=transform)
    test_dataset = datasets.ImageFolder(root='data/testing', transform=transform)    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Instantiate model, loss function, optimizer
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and Evaluate the model
    num_epochs = 100 # 2 # 10
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    evaluate_model(model, test_loader)
    
if __name__ == '__main__':
    time_start = time.time()

#display_setup()
main()

duration = time.time() - time_start

print('\n    running time:  {} (h:m:s)\n'.format(str(datetime.timedelta(seconds=duration))))