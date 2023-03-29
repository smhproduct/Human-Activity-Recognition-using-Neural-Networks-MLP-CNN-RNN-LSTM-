import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
                 hidden_size:int):
        super(MLP, self).__init__()
            
        self.layer1 = nn.Linear(in_features=561, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size//2)
        self.layer3 = nn.Linear(in_features=hidden_size//2, out_features=hidden_size//4)
        self.layer_out = nn.Linear(in_features=hidden_size//4, out_features=6)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x= self.relu(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x)

        return x



class CNN(nn.Module):
    def __init__(self,
                  hidden_size:int):
        super(CNN, self).__init__()
            
        kernel_1 = 5
        padding_1= (kernel_1-1)//2
        kernel_2 = 3
        padding_2= (kernel_2-1)//2
        self.layer1 = nn.Conv1d(in_channels=561, 
                                out_channels=hidden_size, 
                                kernel_size=kernel_1, 
                                padding=padding_1)
        self.layer2 = nn.Conv1d(in_channels=hidden_size, 
                                out_channels=hidden_size//8, 
                                kernel_size=kernel_1, 
                                padding=padding_1)
        self.layer_out = nn.Conv1d(in_channels=hidden_size//8, 
                                    out_channels=6, 
                                    kernel_size=kernel_2, 
                                    padding=padding_2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.layer1(x)
        x= self.relu(x)
        x = self.layer2(x)
        x= self.layer_out(x)
        x = x.transpose(1,2)
        return x
    

class Net(nn.Module):
    def __init__(self,
                  hidden_size:int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(561, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(10, 6)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = x.transpose(1,2)
        return x

class RNN(nn.Module):
    def __init__(self, 
                 hidden_dim:int):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(561, hidden_dim, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 6)
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class LSTM(nn.Module):
    def __init__(self, 
                 hidden_dim:int):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size=561, hidden_size= hidden_dim, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 6)
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        # Initializing hidden state for first input using method defined below
        hidden,cell = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden,cell) = self.lstm(x, (hidden,cell))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden,cell
