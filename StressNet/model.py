import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F

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
    
class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeformConvBlock, self).__init__()
        self.deform_conv_1_1 = DeformConvLayer(in_channels, out_channels)
        self.deform_conv_1_2 = DeformConvLayer(in_channels, out_channels)
        self.deform_conv_1_3 = DeformConvLayer(in_channels, out_channels)
        self.deform_conv_1_4 = DeformConvLayer(in_channels, out_channels)

        self.deform_conv_2_1 = DeformConvLayer(in_channels, out_channels)
        self.deform_conv_2_2 = DeformConvLayer(in_channels, out_channels)

        self.deform_conv_3 = DeformConvLayer(in_channels, out_channels)

    def forward(self, x):
        inputs = x
        out_1 = self.deform_conv_1_1(inputs)
        out_1 = self.deform_conv_1_2(out_1)

        out_2 = self.deform_conv_2_1(inputs)

        out_sum_1 = out_1 + out_2

        out_1 = self.deform_conv_1_3(out_sum_1)
        out_1 = self.deform_conv_1_4(out_1)

        out_2 = self.deform_conv_2_2(out_sum_1)

        out_3 = self.deform_conv_3(out_sum_1)

        return out_1 + out_2 + out_3
    
class MDCN(nn.Module):
    def __init__(self, in_channels):
        super(MDCN, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=0, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=0, stride=1),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.deform_block1 = DeformConvBlock(128, 128)
        self.deform_block2 = DeformConvBlock(128, 128)
        self.deform_block3 = DeformConvBlock(256, 256)
        self.deform_block4 = DeformConvBlock(256, 256)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.downsample(x)
        x = self.conv1(x)
        #print(f"conv 1 size: {x.size()}")
        x = self.conv2(x)
        #print(f"conv 2 size: {x.size()}")
        x = self.conv3(x)
        #print(f"conv 3 size: {x.size()}")
        x = self.deform_block1(x)
        #print(f"Deform Conv 1 size: {x.size()}")
        x = self.deform_block2(x)
        #print(f"Deform Conv 2 size: {x.size()}")
        x = self.conv4(x)
        #print(f"conv 4 size: {x.size()}")
        x = self.deform_block3(x)
        #print(f"Deform Conv 3 size: {x.size()}")
        x = self.deform_block4(x)
        #print(f"Deform Conv 4 size: {x.size()}")
        return x.view(B, -1)
    
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out
    
class ARCNN(nn.Module):
    def __init__(self, num_classes):
        super(ARCNN, self).__init__()

        self.num_classes = num_classes

        self.mdcn_model = MDCN(5)
        
        self.embedding_size = 256 * 3 * 3

        # Attention layer
        self.attention = AttentionLayer(embed_dim=1024, num_heads=8)

        # Recurrent layer (LSTM)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=512,
                            num_layers=1, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(2*512, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (T, B, C, H, W)
        B, T, C, H, W = x.size()
        mdcn_outputs = []
        for i in range(T):
            out = self.mdcn_model(x[:, i, :, :, :])
            #out = out.view(out.size(0), -1)
            out = out.unsqueeze(0)
            mdcn_outputs.append(out)

        # Concatenate the outputs from the mdcn models
        # (13, B, 256x3x3)
        out = torch.cat(mdcn_outputs, dim=0)
        
        # Reshape the output for attention layer
        #out = out.permute(0, 2, 1)  # shape: (batch_size, channels, seq_len)
        #print("\nMDCN Output:",out.size())
        
        # Reshape for LSTM input
        out = out.permute(1, 0, 2)  # shape: (batch_size, seq_len, channels)

        # Recurrent layer (Bi-LSTM)
        out, _ = self.lstm(out)
        #print("\nBi-LSTM Output:",out.size())
        
        # Apply attention layer---------------------------
        out = self.attention(out)
        #print("\nAttention Output:",out.size())
        # Reshape the output tensor to (batch_size, output_size)
        #out = out.permute(0, 2, 1).reshape(64, 1024)
        #out = out.mean(dim=1)
        #print("\nSoftmax Input:",out.size())
        #-------------------------------------------------
        
        # Fully connected layers
        out = F.relu(self.fc1(out[:, -1, :]))  # Take the last timestep output
        out = self.fc2(out)

        return F.softmax(out, dim=1)

if __name__ == '__main__':
    inputs = torch.randn(64, 13, 5, 140, 140)
    model = ARCNN(num_classes=3)
    out = model(inputs)
    # returns the output probabilities of Softmax Layer; '3' is the number of categories
    out.size()