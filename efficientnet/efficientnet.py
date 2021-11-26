from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
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


class EfficientNet_b1(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b1, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
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
            nn.Linear(1408, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
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


class EfficientNet_b3(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.classifier = nn.Sequential(
            nn.Linear(1536, 718, bias=True),
            nn.BatchNorm1d(718),
            nn.Dropout(0.2),
            nn.Linear(718, 359, bias=True),
            nn.BatchNorm1d(259),
            nn.Dropout(0.2),
            nn.Linear(359, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_b4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.classifier = nn.Sequential(
            nn.Linear(1792, 896, bias=True),
            nn.BatchNorm1d(896),
            nn.Dropout(0.2),
            nn.Linear(896, 448, bias=True),
            nn.BatchNorm1d(448),
            nn.Dropout(0.2),
            nn.Linear(448, 224, bias=True),
            nn.BatchNorm1d(224),
            nn.Dropout(0.2),
            nn.Linear(224, num_classes, bias=True),
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
