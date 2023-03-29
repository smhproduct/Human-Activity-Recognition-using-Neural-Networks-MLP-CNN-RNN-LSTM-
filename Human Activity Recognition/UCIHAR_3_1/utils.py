import pandas as pd
import os
from sklearn.utils import shuffle
import torch

def feature_analysis(root: str):

    features = []
    with open(os.path.join(root,"features.txt")) as file:
        for line in file:
            features.append(line.split()[1])
                
    # SERIALIZING DUPLICATE FEATURE NAMES
    names = []
    count = {}
    for feature in features:
        if(features.count(feature) > 1):
            names.append(feature)
    for name in names:
        count[name] = features.count(name)
    
    for i in range(len(features)):
        if(features[i] in names):
            num = count[features[i]]
            count[features[i]] -= 1;
            features[i] = str(features[i] + str(num))
    return features



def load_data(root:str, 
              features:list, 
              train:bool=True):
    if train == True: 
        dataset_type = 'train'
    else: 
        dataset_type = 'test'
        
    raw_data = pd.read_csv(os.path.join(root, dataset_type, f"X_{dataset_type}.txt"), 
                           delim_whitespace = True,
                           names = features)
    
    raw_data['subject_id'] = pd.read_csv(os.path.join(root, dataset_type, f"subject_{dataset_type}.txt"),header=None,squeeze=True)
    raw_data['activity'] = pd.read_csv(os.path.join(root, dataset_type, f"y_{dataset_type}.txt"),header=None,squeeze=True)
    activity = pd.read_csv(os.path.join(root, dataset_type, f"y_{dataset_type}.txt"),header=None,squeeze=True)
    label_name = activity.map({1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"})
    raw_data["activity_name"] = label_name
    raw_data['activity'] -= 1
    # Storing data into a csv file
    # raw_data.to_csv(os.path.join(root,dataset_type,f"{dataset_type}.csv"), index= False)
    
    return raw_data


def preprocessing(raw_data):
    # SHUFFLING DATA
    shuffled_data = shuffle(raw_data)
    shuffled_data= shuffled_data.reset_index(drop=True)
    
    
    # SEPERATING LABEL
    #print(list(shuffled_data.columns.values))
    label = shuffled_data[['activity']]
    # print("LABELSSSSS",label)
    
    # DROP COLUMNS WITH METADATA
    preprocessed_data = shuffled_data.drop(['activity','activity_name','subject_id'], axis=1)
    #print("PRESPROCESSED DATA\n", preprocessed_data)
      
    return preprocessed_data, label


def windowing(data,window_size,steps):
    rows = data.shape[0]
    
    batches=[]
    for i in range(0, rows, steps):
        if (i+window_size)<rows:
            windowed_data=torch.FloatTensor(data.iloc[i:i+window_size].values)
            windowed_data = windowed_data.unsqueeze(0)
            batches.append(windowed_data)
    
    batched_data = torch.cat(batches,dim = 0)
    return batched_data
    

    
    
    
    
    
    