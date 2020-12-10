import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import random as rdm

class LSTM(nn.Module):
    """
    pytorch to define LSTM network


    """
 
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, num_directions*hidden_size]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view( self.batch_size, len(input[0]), -1), self.hidden)
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return y_pred.view(-1)
 

def get_training_data(data_ID, nn):
    """
    data_ID ... filename id for data to be loaded
    nn ... number of nearest neighbours used in the dataset
    
    returns x as the inputs to the network and y as the truth values
    """

    
    #container for loaded data
    
    data = np.genfromtxt("./training_data/training_set-" + data_ID, delimiter = ",")
    #retrieve batch_size and seq_len
    batch_size = len(data)
    seq_len = len(data[0])//(2*nn+2)
    #tensor with input data = neighbours positions
    x = np.zeros((batch_size,seq_len,2*nn))
    #tensor with prediction data = position of agent at next timestep
    y = np.zeros((batch_size, seq_len,2))
    #fill tensors
    for j in range(seq_len):
        x[:,j,:] = data[:,j*(2*nn+2):(j+1)*2*nn+2*j]
        y[:,j,:] = data[:,(j+1)*2*nn+2*j:(j+1)*2*nn+2*j+2]
    #for pytorch nn model we need tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x,y


def main():
        
    dtype = torch.float
    device = torch.device("cpu") #only training with cpu was possible on our computers

    data_ID = "180-140-5"
    nene = 5

    filename_output = "LSTM__ID-" + data_ID + ".model" 

    
    #x is input data, y is target data
    x,y = get_training_data(data_ID,nene)

    batch_size = len(x)
    output_dim = 2
    input_dim = len(x[0,0])
    num_layers = 3
    h1 = 20
    seq_len = len(x[0])

    mini_batch_size = batch_size

    model = LSTM(input_dim, h1, batch_size=mini_batch_size, output_dim=output_dim, num_layers=num_layers)
    model.train()


    lr = 1e-2
    num_epochs = 1000000

    loss_fn = torch.nn.MSELoss(size_average=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)



    
    #####################
    # Train model
    #####################
    
    hist = np.zeros(num_epochs)

    quit_timer = 0
    tol = 0.01

    x_mini = x
    y_mini = y


    print("MSE of nothing: ", loss_fn(torch.zeros_like(y_mini), y_mini).item())
    
    for t in range(num_epochs):
        # Clear stored gradient
        model.zero_grad()

        #uncomment this to enable batching
        #take random mini-batch
       # if t%5000==0:
       #     batches_to_take = rdm.sample(range(0,batch_size),mini_batch_size)
       #     x_mini = x[batches_to_take]
       #     y_mini = y[batches_to_take]
        
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()
        
        y_pred = torch.zeros_like(y_mini)
        for a in range(seq_len):
            y_pred[:,a] = model(x_mini[:,a].unsqueeze(1)).view(mini_batch_size,output_dim)
        
        loss = loss_fn(y_pred, y_mini)
        hist[t] = loss.item()
        if t % 50 == 0:
            #print("Epoch ", t, "MSE: ", loss.item())
            print("Epoch ", t, "MSE: ", loss.item())
            epochs = range(1,t+2)
            plt.plot(epochs, np.log10(hist[:t+1]), 'r')

            #plt.hlines(y=7, xmin = epochs[0], xmax = epochs[-1], linestyle="--")
            plt.savefig("progress.png")
            plt.close()
            torch.save(model, filename_output)

        #
        # adapt learning speed
        #
            
        if hist[t]<1e1 and not lr <= 1e-3:
            print("switched to lr=1e-3")
            lr = 1e-3
            optimiser = torch.optim.Adam(model.parameters(), lr)
        if hist[t]<1e0 and not lr <= 1e-4:
            print("switched to lr=1e-4")
            lr = 1e-4
            optimiser = torch.optim.Adam(model.parameters(), lr)

        if hist[t]<1e-1 and not lr <= 1e-5:
            print("switched to lr=1e-5")
            lr = 1e-5
            optimiser = torch.optim.Adam(model.parameters(), lr)

        

        
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()
    
        # Backward pass
        loss.backward()
    
        # Update parameters
        optimiser.step()



        if hist[t-1]-hist[t]<tol:
            quit_timer += 1
        else:
            quit_timer = 0
        if quit_timer>500:
            print("early stopping, steady state reached")
            break

    print("Training complete, enjoy your model!")
    torch.save(model, filename_output)


main()
