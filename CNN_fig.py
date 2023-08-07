import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
class CNNIQAnet(nn.Module):
    def __init__(self, patch_size):
        super(CNNIQAnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(patch_size * patch_size * 128, 128)
        self.fc2 = nn.Linear(128, 1)

        self.patch_size = patch_size

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        conv8 = F.relu(self.conv8(x))

        # Global Average Pooling
        q = F.adaptive_avg_pool2d(conv8, (1, 1))
        q = q.view(-1, self.patch_size * self.patch_size * 128)

        q = F.relu(self.fc1(q))
        q = self.fc2(q)

        return q

# Create an instance of the model
model = CNNIQAnet(patch_size=32)

# Create a random input tensor (example: batch size of 1 with 1 channel and 32x32 image size)
input_data = torch.randn(1, 1, 32, 32)

# Generate a graph of the model architecture
output = model(input_data)
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph to a file (optional)
graph.render(filename='C:/Users/win 10/Desktop/optim_CNNIQA/cnn_iqa_net', format='png')
