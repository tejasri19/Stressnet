# Base: VGG-16 + 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops

class DeformConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformConvLayer, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding, stride=stride)
        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset = self.offset_conv(x)
        weight = self.deform_conv.weight
        x = ops.deform_conv2d(x, offset, weight, padding=self.offset_conv.padding[0])  # Adjust padding
        x = self.norm(x)
        x = F.relu(x)
        return x
    

class ARCNN(nn.Module):
    def __init__(self, num_classes, n_days):
        super(ARCNN, self).__init__()

        self.num_classes = num_classes
        self.n_days = n_days


        self.alexnet = models.alexnet(pretrained=False)


        # Load the pretrained VGG model
        self.vgg = models.vgg16(pretrained=False)

        # Remove the fully connected layers
        self.vgg.classifier = torch.nn.Identity()

        # Modify the input layer to accept 5 channels
        self.vgg.features[0] = torch.nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.features[28] = DeformConvLayer(512, 512)


        self.base_model = self.vgg
        #(64, 512, 4, 4)
        
        self.embedding_size = 512 * 4 * 4

        # Recurrent layer (LSTM)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=512,
                            num_layers=1, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(2*512, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (Batch_size, Time, Channel, Height, Width)
        B, T, C, H, W = x.size()
        base_outputs = []
        for i in range(T):
            out = self.base_model.features(x[:, i, :, :, :]) # torch.Size([64, 5, 140, 140]): x[:, i, :, :, :]
            out = out.view(out.size(0), -1)
            out = out.unsqueeze(0)
            base_outputs.append(out)

        out = torch.cat(base_outputs, dim=0)
        
        out = out.permute(1, 0, 2)  # shape: (batch_size, seq_len, channels)

        out, _ = self.lstm(out)
        out = F.relu(self.fc1(out[:, -1, :]))  # Take the last timestep output, ie, [64, 1024]
        out = self.fc2(out)

        return F.softmax(out, dim=1)