"""
Created on Tue Sep 25 14:32:06 2018

@author: kavjit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


import numpy as np
import h5py
from random import randint
import time 


#loading data
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=0)		#numworkers?

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):										#why nnmodule?
    def __init__(self):
        super(Net, self).__init__() 						#whats this?
        self.conv1 = nn.Conv2d(3, 64, 4, padding = 2, stride = 1) 		#correcT?
        self.conv2 = nn.Conv2d(64, 64, 4, padding = 2, stride = 1)
        self.conv3 = nn.Conv2d(64, 64, 4, padding = 2, stride = 1)
        self.conv4 = nn.Conv2d(64, 64, 4, padding = 2, stride =1)
        self.conv5 = nn.Conv2d(64, 64, 4, padding = 2, stride =1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride =1, padding = 0)
        self.conv7 = nn.Conv2d(64, 64, 3, stride =1, padding = 0)
        self.conv8 = nn.Conv2d(64, 64, 3, stride =1, padding = 0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3) 	#2d?  
        self.fc1 = nn.Linear(1024, 500) 					#correct? 
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.batchnorm(x)
        x = F.relu(self.conv2(x))

        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))

        x = self.batchnorm(x)
        x = F.relu(self.conv4(x))

        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv5(x))

        x = self.batchnorm(x)
        x = F.relu(self.conv6(x))

        x = self.dropout(x)
        x = F.relu(self.conv7(x))

        x = self.batchnorm(x)
        x = F.relu(self.conv8(x))

        x = self.batchnorm(x)
        x = self.dropout(x)

        x = x.view(-1, 1024)	#??????	
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.size())
        #soft = nn.Softmax(dim=1) #correct?
        #x = soft(x)
        return x

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)


#hyperparameters
epochs = 200

for epoch in range(epochs):  
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) #why?
        correct += (predicted == labels).sum().item()
    
    print('accuracy at epoch {} = {}'.format(epoch,correct/total))


print('Finished Training')



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
    #for i,data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)   #output.data? #_for ignoring a value while unpacking second return is argmax index, 1 to indicate axis to check
        total += labels.size(0) #why?
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))








