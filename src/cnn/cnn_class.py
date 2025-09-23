import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*157*157, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print("After conv1+pool:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("After conv2+pool:", x.shape)
        x = torch.flatten(x, 1)
        #print("After flatten:", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_loss():
    return nn.CrossEntropyLoss()

def get_optimizer(model, lr=0.001, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# if __name__ == "__main__":
#     net = Net(num_classes=36)
#     criterion = get_loss()
#     optimizer = get_optimizer(net)
#     print("Model, loss i optimizer su inicijalizovani.")