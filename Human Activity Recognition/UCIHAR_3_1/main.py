import dataset
import os 
import pathlib
import networks
import trainer
import plotting
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


main_folder = pathlib.Path(__file__)
parent_folder= main_folder.parents[1]
data_folder = os.path.join(parent_folder, "Datasets", "UCI HAR Dataset")

window_size=40
steps=15
batch_size=64
hidden_size =512
lr=0.001
epochs = 25


net = 'LSTM'

train_dataset = dataset.UCIHARDataset(root= data_folder, 
                                      window_size = window_size,
                                      steps= steps,
                                      train= True)

test_dataset = dataset.UCIHARDataset(root= data_folder, 
                                      window_size = window_size,
                                      steps= window_size,
                                      train= False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model=networks.LSTM(hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
# for name,params in model.named_parameters():
#     print(name,params.shape)

# TRAINING
train_losses = []
train_accuracies = []
for epoch in range(epochs):
    train_loss, train_accuracy = trainer.training_loop(train_loader, optimizer, model, criterion)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch} =>\ttrain_accuracy = {train_accuracy}\ttrain_loss = {train_loss}")
    
plotting.plot_losses(train_losses, net)
plotting.plot_accuracies(train_accuracies, net)

#TESTING
test_loss, test_accuracy, test_pred, test_true = trainer.testing_loop(test_loader, model, criterion)
print(f"\n\n{net} model\nACCURACY: {test_accuracy} % \t\tLOSS: {test_loss} ")    

plotting.Confusion_Matrix(sum(test_true, []), sum(test_pred, []), net)

# 2947