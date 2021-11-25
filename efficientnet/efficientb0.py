from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNet_b2(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b2, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        self.classifier = nn.Sequential(
            nn.Linear(1408, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNet_b5(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes, bias=True),
            nn.Sigmoid()

        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier(x)
        return x
