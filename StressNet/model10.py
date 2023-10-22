# Base:    VGG-16 + DeformConvLayer + Weighted Attention


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.models as models

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

# Original
class ARCNN(nn.Module):
    def __init__(self, num_classes, n_days):
        super(ARCNN, self).__init__()

        self.num_classes = num_classes
        self.n_days = n_days

        # Load the pretrained VGG model
        self.vgg = models.vgg16(pretrained=False)

        # Remove the fully connected layers
        self.vgg.classifier = torch.nn.Identity()

        # Modify the input layer to accept 5 channels
        self.vgg.features[0] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.features[28] = DeformConvLayer(512, 512)
        
        self.base_model = self.vgg
        #(64, 512, 4, 4)

        self.embedding_size = 512 * 4 * 4

        # Recurrent layer (LSTM)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=512,
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # Weighted Attention Layer
        self.weights = nn.Parameter(torch.Tensor(1, 13))
        nn.init.kaiming_normal_(self.weights)
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
        #print(f'Feature Extractor Output:{out.shape}')

        out, _ = self.lstm(out)
        #print(f'LSTM Output:{out.shape}')
        
        H = torch.tanh(out)
        #print(f'H size: {H.shape}')
        alpha = F.softmax(torch.matmul(self.weights, H).squeeze(), dim=1)
        #print(f'weights size: {self.weights.shape}')
        alpha = alpha.unsqueeze(2)
        R = torch.matmul(H, alpha)
        R = R.permute(0, 2, 1)
        out = torch.matmul(R, out)

        out = F.relu(self.fc1(out.squeeze()))
        out = self.fc2(out)

        return F.softmax(out, dim=1)