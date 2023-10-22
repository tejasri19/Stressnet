# Base: AlexNet + Weighted Attention

# Base: AlexNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Original
class ARCNN(nn.Module):
    def __init__(self, num_classes):
        super(ARCNN, self).__init__()

        self.num_classes = num_classes

        self.alexnet = models.alexnet(pretrained=False)


        # Modify the first layer to accept 5-channel input
        self.alexnet.features[0] = torch.nn.Conv2d(5, 64, kernel_size=11, stride=4, padding=2)

        # Remove the fully connected layers from the classifier
        self.alexnet.classifier = torch.nn.Identity()

        self.base_model = self.alexnet
        #(64, 256, 3, 3)
        
        self.embedding_size = 256 * 3 * 3

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

        out, _ = self.lstm(out)
        H = torch.tanh(out)
        alpha = F.softmax(torch.matmul(self.weights, H).squeeze(), dim=1)
        alpha = alpha.unsqueeze(2)
        R = torch.matmul(H, alpha)
        R = R.permute(0, 2, 1)
        out = torch.matmul(R, out)

        out = F.relu(self.fc1(out.squeeze()))
        out = self.fc2(out)

        return F.softmax(out, dim=1)