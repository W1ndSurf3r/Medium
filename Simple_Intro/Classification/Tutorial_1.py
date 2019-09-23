from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target 

# You can try nn.Sigmoid(), nn.Tanh(), nn.ELU() instead of nn.ReLU() as explained in the post

model = nn.Sequential(nn.Linear(2,5, bias=True),nn.ReLU(), # adding a fully connected layer with 2 inputs and 5 hidden outputs
                      nn.Linear(5,20, bias=True), nn.ReLU(), # the 5 outputs are now inputs to the next layer with 20 hidden outputs
                      nn.Linear(20,3, bias=True)) # finally, the 20 hidden outputs to the number of classes 3
                             

optimizer = optim.Adam(model.parameters(),lr=0.01) # defining the optimizer as explained in part 3
print(model) # printing our model


train_X = torch.from_numpy(X).float() # converting our numpy array to torch tensors then to float
train_y = torch.from_numpy(y).long() # converting our numpy array to torch tensors and labels to type long

loss_fn = nn.CrossEntropyLoss() # Define loss function, part 2 

for i in range(5000): # training for 5000 epochs
    optimizer.zero_grad() # don't worry about this now but its resetting the gradients values from previous epoch
    output = model(train_X) # feedforward 
    loss = loss_fn(output, train_y) # calculating the loss (the error)
    loss.backward() # backpropagation to calculate the derivates with respect to the loss
    optimizer.step() # adjusting the weights 
    
# VERY IMPORTANT:: The following is to test the accuracy. This is not the correct way, this can be the training accuracy 
# but it is not how the models are evaluated. We will cover that, but for now you can check your model performance this way.

outputs = model(train_X) # feed forward
_, predicted = torch.max(outputs.data, 1) # we get the class of the highest probability 
y_hat = predicted.detach().numpy() # return to numpy array

train_accuracy  = accuracy_score(y, y_hat) # the accuracy function calculated the number of the correct predicitions
print(train_accuracy)
