import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class PneumoniaCNN(nn.Module):

    def __init__(self, config:Config):
        super(PneumoniaCNN, self).__init__()

        self.config = config

        self.conv1 = nn.Conv2d(config.in_channels, 32, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32) if config.use_batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias = False )
        self.bn2 = nn.BatchNorm2d(64) if config.use_batch_norm else nn.Identity()

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias = False )
        self.bn3 = nn.BatchNorm2d(128) if config.use_batch_norm else nn.Identity()

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias = False )
        self.bn4 = nn.BatchNorm2d(256) if config.use_batch_norm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, config.num_classes)

        self.dropout = nn.Dropout(p = config.dropout_rate)
        #self.dropout2 = nn.Dropout2d(0.5)

        self._initialize_weights()

    def _initialize_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace = True)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace = True)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace = True)
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x