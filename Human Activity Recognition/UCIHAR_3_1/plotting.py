import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Create directory for storing plots
main_folder = pathlib.Path(__file__)
parent_folder= main_folder.parents[1]
plot_folder = os.path.join(parent_folder, "UCIHAR_3_1", "Results Visualization")
pathExists = os.path.exists(plot_folder)
if not pathExists:
    os.mkdir(plot_folder)

def activity_visualizing(activity_name, train=True): 
    if train == True: 
        dataset_type = 'train'
    else: 
        dataset_type = 'test'
        
    activity_name = list(activity_name)
    activity_freq={}
    for items in activity_name:
        activity_freq[items] = activity_name.count(items)
    #Extracting x and y coordinates
    Activities = activity_freq.keys()
    Counts = activity_freq.values()
    
    fig = plt.figure(figsize = (10, 5))
    plt.bar(Activities, Counts, color ='maroon',
        width = 0.4)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title(f"Count of each activity ({dataset_type} data set)")
    plt.tick_params(labelsize = 8)
    plt.xticks(rotation = 45)
    file_name = f'Activity visualization_{dataset_type}.png'
    fig.savefig(os.path.join(plot_folder, file_name))
    plt.show()
    


def plot_losses(train_losses: list, net: str):
    fig = plt.figure(figsize = (10, 5))
    plt.plot(train_losses)
    plt.xlabel("No. of epochs")
    plt.ylabel("Training Loss")
    plt.title(f"{net} Loss function Plot")
    plt.show()
    file_name = f'{net} model Training Loss Curve.png'
    fig.savefig(os.path.join(plot_folder, file_name))
    
def plot_accuracies(train_accuracies: list, net:str):
    fig = plt.figure(figsize = (10, 5))
    plt.plot(train_accuracies)
    plt.xlabel("No. of epochs")
    plt.ylabel("Training Accuracy")
    plt.title(f"{net} Accuracy Plot")
    plt.show()
    file_name = f'{net} model Training Accuracy Curve.png'
    fig.savefig(os.path.join(plot_folder, file_name))
    
def Confusion_Matrix(test_true: list, test_pred: list, net:str):
    confusionMatrix = confusion_matrix(test_true, test_pred)
    sns.set(font_scale=1.5)
    labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LYING"]
    fig=plt.figure(figsize=(18,9))
    sns.heatmap(confusionMatrix, cmap = "Blues", annot = True, fmt = ".0f", xticklabels=labels, yticklabels=labels)
    plt.title(f"{net} Model Confusion Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 45)
    plt.show()
    file_name = f'{net} model Confusion Matrix.png'
    fig.savefig(os.path.join(plot_folder, file_name))