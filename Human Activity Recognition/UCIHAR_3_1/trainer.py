# -*- coding: utf-8 -*-
import torch
def training_loop(training_loader, optimizer, model, criterion):
    running_loss = 0.0
    running_acc = 0.0
    
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero the gradients for every batch
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        
        #print(outputs.shape, labels.shape)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        labels = labels.reshape(-1).long()
        #print(outputs.shape, labels.shape)
        # But the labels must be an INTEGER value containing the best index.
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        outputs = torch.argmax(outputs, dim = 1)
        correct = torch.sum(outputs == labels).item()
        accuracy = (correct / outputs.shape[0]) * 100
        
        running_acc += accuracy
        
    avg_loss= running_loss/(i+1)
    avg_accuracy = running_acc / (i+1)
    return avg_loss, avg_accuracy


def testing_loop(test_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        running_loss =0.0
        running_acc = 0.0
        
        test_pred = []
        test_true = []
        
        for i,(inputs, labels) in enumerate(test_loader):
            
            outputs = model(inputs)
            
            outputs = outputs.reshape(-1, outputs.shape[-1])
            labels = labels.reshape(-1).long()
            
            loss = criterion(outputs,labels)
            running_loss += loss.item()
            
            outputs = torch.argmax(outputs, dim=1)
            
            
            correct = torch.sum(outputs == labels).item()
            accuracy = (correct/outputs.shape[0]) * 100
            
            
            running_acc += accuracy
            # print(i, accuracy)
            
            test_pred.append(outputs.tolist())
            test_true.append(labels.tolist())

        avg_loss = running_loss/(i+1)
        avg_acc = running_acc/(i+1)
        
    return avg_loss, avg_acc, test_pred, test_true

# GRADIENT: Change in weight with respect to change in error