import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

# Hyperparameters
epochs = 3
batch_size = 4 
hidden_size = 500

# I/O data
train_data = torchvision.datasets.MNIST('./data',train=True,transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST('./data',train=False,transform=transforms.ToTensor())

train_load = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
test_load = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=8)

# CPU / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model class
class ffNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(ffNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.z1  = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.z1(out)
        out = self.fc2(out)
        return out

# Define network
net = ffNet(28*28, 24, 10)
# Defile loss
criterion = torch.nn.CrossEntropyLoss()
# Define optimizer
optim = torch.optim.Adam(net.parameters(), lr=0.001)

# Training
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_load):
        # Send I/O to devices
        x = x.to(device)
        y = y.to(device)

        # Zero parameter gradients
        optim.zero_grad()
        
        # Forward
        yhat = net(x.view(-1, 28*28))
        loss = criterion(yhat, y)

        # Calculate gradient and update weights
        loss.backward()
        optim.step()
        
        if i % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, 4, i+1, len(train_data)//batch_size, loss.data[0]))

# Testing
correct = 0
for i, (x,y) in enumerate(test_load):
    _, yhat = torch.max(net(x.view(-1, 28*28)).data, 1)
    correct += (y == yhat).sum()
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100 * correct / 10000))

# Save trained model
torch.save(net.state_dict(), 'model.pkl')
