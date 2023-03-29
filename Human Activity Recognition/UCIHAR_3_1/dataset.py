from torch.utils.data import Dataset
import utils
import plotting
import networks

class UCIHARDataset(Dataset):
    
    def __init__(self,
                  root: str,
                  window_size:int,
                  steps:int,
                  train: bool = True):
        features = utils.feature_analysis(root = root)
        #print(features) 
        #tGravityAccMag-arCoeff()1
        raw_data = utils.load_data(root = root,
                                    features = features, 
                                    train = train)
        plotting.activity_visualizing(activity_name = raw_data['activity_name'],
                                                          train=train)
        preprocessed_data, label = utils.preprocessing(raw_data=raw_data)
        self.batched_data=utils.windowing(preprocessed_data,window_size,steps)
        self.batched_label =utils.windowing(label,window_size,steps)
        self.batched_label =  self.batched_label.squeeze(-1)
    def __len__(self):
        return len(self.batched_data)
    
    def __getitem__(self, index):
        return self.batched_data[index], self.batched_label[index]
        
            
   