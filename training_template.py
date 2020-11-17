import torch
import matplotlib.pyplot as plt
import numpy as np




dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size (number of datasets); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 1, 500, 1

#create sine data
#change this to import data from datasets
x = torch.linspace(0, 10*np.pi, N, device=device, dtype=dtype)
y = torch.sin(x)

#just reshaping - remove this
x = x.reshape((1,N)).T
y = y.reshape((1,N)).T



#define network
model = torch.nn.Sequential(

    #linear layer
    torch.nn.Linear(D_in, H),
    
    #apply nonlinearity also works with torch.nn.ReLU()
    torch.nn.Tanh(),

    torch.nn.Linear(H, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

#choose loss function - can be played around with
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(5000):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = model(x)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())


    model.zero_grad()
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad


#plotting of sine wave
#change this to whatever necessary
x_test = torch.linspace(0, 8*np.pi, 100).reshape(1,100).T
y_test = model(x_test)

plt.plot(x_test.detach().numpy(),y_test.detach().numpy())
plt.plot(x.detach().numpy(),y.detach().numpy())

plt.show()
