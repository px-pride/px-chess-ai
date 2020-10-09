import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels (RGB), 6 output channels
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)

        # 6 input filter channels, 16 output channels
        # 3x3 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)

        # affine operation
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # first conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        # second conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        # reshape x for fc layer
        x = x.view(-1, self.num_flat_features(x))

        # fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# build network, loss function, optimizer
net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9)

# build (random) dataset
#random_input = torch.randn(8, 1, 32, 32)
#random_target = torch.rand(8, 10)

# build actual dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)
testloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=False,
    num_workers=0)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)

print(torch.cuda.is_available())
device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
net.to(device)


# training loop
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = net(inputs) # forward pass
        loss = loss_fn(output, targets)
        loss.backward(retain_graph=True) # backward pass
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# batch is a tensor of size [batch_size, x_res, y_res, channels]
# affine: hidden = inputs * weights + bias