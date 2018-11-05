import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

# Hyperparameters
epochs = 3
batch_size = 4 

## Model parameters
num_filter = 16
filter_size = 3
stride_size = 1
padding_size = 1
max_pool_size = 2

# I/O data
train_data = torchvision.datasets.MNIST('./data',train=True,transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST('./data',train=False,transform=transforms.ToTensor())

train_load = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
test_load = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=8)

# CPU / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model class
class ffNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ffNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filter, filter_size, stride_size, padding_size) # O: 28 x 28 x 16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(max_pool_size) # O: 14 x 14 x 16
   
        self.conv2 = nn.Conv2d(num_filter, num_filter*2, filter_size, stride_size, padding_size) # O: 14 x 14 x 32
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(max_pool_size) # O: 7 x 7 x 32
        self.line1 = nn.Linear(7*7*32, output_size)
  
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.line1(out)
        return out

# Define network
net = ffNet(28*28, 10)
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
        yhat = net(x)
        loss = criterion(yhat, y)

        # Calculate gradient and update weights
        loss.backward()
        optim.step()
        
        if i % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, 4, i+1, len(train_data)//batch_size, loss.data[0]))

# Testing
correct = 0
for i, (x,y) in enumerate(test_load):
    _, yhat = torch.max(net(x).data, 1)
    correct += (y == yhat).sum()
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100 * correct / 10000))

# Save trained model
torch.save(net.state_dict(), 'model.pkl')
