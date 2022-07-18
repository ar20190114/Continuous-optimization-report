import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim
import matplotlib.pyplot as plt


# transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# dataset
train_dataset = MNIST(root='./content/',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = MNIST(root='./content/',
                              train=False,
                              transform=transform)

# Data Loader
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#make CNN
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


#make graph
def PlotGraph(title, ylabel, y_min, y_max, values_sgd, values_adagrad, values_rms, values_adadelta, values_adam, values_adamw, rng, label_sgd, label_adagrad, label_rms, label_adadelta, label_adam, label_adamw):
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.ylim(y_min, y_max)
    plt.plot(range(rng), values_sgd, label=label_sgd)
    plt.plot(range(rng), values_adagrad, label=label_adagrad)
    plt.plot(range(rng), values_rms, label=label_rms)
    plt.plot(range(rng), values_adadelta, label=label_adadelta)
    plt.plot(range(rng), values_adam, label=label_adam)
    plt.plot(range(rng), values_adamw, label=label_adamw)
    plt.legend()
    plt.grid()
    plt.show()


classes = ['SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW']
loss_cnn = {}
acc_cnn = {}
for key in classes:
    loss_cnn[key] = []
    acc_cnn[key] = []

lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizers = {}

for key in classes:

    model = Cnn()

    optimizers['SGD'] = optim.SGD(model.parameters(), lr)
    optimizers['Adagrad'] = optim.Adagrad(model.parameters(), lr)
    optimizers['RMSprop'] = optim.RMSprop(model.parameters(), lr)
    optimizers['Adadelta'] = optim.Adadelta(model.parameters(), lr)
    optimizers['Adam'] = optim.Adam(model.parameters(), lr)
    optimizers['AdamW'] = optim.AdamW(model.parameters(), lr)

    for epoch in range(50):
        model.train()

        running_loss = 0.0
        for i, (image, label) in enumerate(train_loader):
            optimizers[key].zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizers[key].step()
            loss_cnn[key].append(loss.item())

            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/100))

        model.eval()
        with torch.no_grad():
            for t_image, t_label in test_loader:
                output = model(t_image)
                _, predicted = torch.max(output, 1)
                class_correct = (predicted == t_label).sum().item()
                acc = class_correct / len(predicted) * 100
                acc_cnn[key].append(acc)

PlotGraph('loss', 'loss', 0, 2, loss_cnn['SGD'], loss_cnn['Adagrad'], loss_cnn['RMSprop'], loss_cnn['Adadelta'], loss_cnn['Adam'], loss_cnn['AdamW'], 30000, 'SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW')
PlotGraph('acc', 'acc', 60, 100, acc_cnn['SGD'], acc_cnn['Adagrad'], acc_cnn['RMSprop'], acc_cnn['Adadelta'], acc_cnn['Adam'], acc_cnn['AdamW'], 30000, 'SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW')
