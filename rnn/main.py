import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

# Hyperparameters
epochs = 2
batch_size = 100

# RNN settings
input_size = 28
seq_size = 28
hidden_size = 128
output_size = 10
num_layers = 2

# I/O data
train_data = torchvision.datasets.MNIST('./data',train=True,transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST('./data',train=False,transform=transforms.ToTensor())

train_load = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
test_load = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=8)

# CPU / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.line = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.hidden = (torch.randn(num_layers, batch_size, hidden_size), torch.randn(num_layers, batch_size, hidden_size))
        out, _ = self.lstm(x, self.hidden)
        out = self.line(out[:,-1,:])
        return out

# Define network
net = LSTM(input_size, hidden_size, num_layers, output_size)
# Defile loss
criterion = torch.nn.CrossEntropyLoss()
# Define optimizer
optim = torch.optim.Adam(net.parameters(), lr=0.01)

# Training
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_load):
        # Send I/O to devices
        x = x.to(device) # 100 images (100, 1, 28, 28)
        y = y.to(device) # 100 labels
        # Zero parameter gradients
        optim.zero_grad()
        
        # Forward
        yhat = net(x.view(batch_size, seq_size, input_size))
        loss = criterion(yhat, y)

        # Calculate gradient and update weights
        loss.backward()
        optim.step()
        
        if i % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, epochs, i+1, len(train_data)//batch_size, loss.data[0]))

# Testing
correct = 0
for i, (x,y) in enumerate(test_load):
    _, yhat = torch.max(net(x.view(batch_size, seq_size, input_size)).data, 1)
    correct += (y == yhat).sum()
print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100 * correct / 10000))

# Save trained model
torch.save(net.state_dict(), 'model.pkl')
