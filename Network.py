from torch import nn
import torch.nn.functional as F
import torch



class CNNIQAnet(nn.Module):
    def __init__(self):
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


        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input):
        x = input.view(-1, input[0].size(-3), input[0].size(-2), input[0].size(-1))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        conv8 = F.relu(self.conv8(x))
        q = torch.nn.functional.adaptive_avg_pool2d(conv8, (1, 1))
        q = q.squeeze(3).squeeze(2)

        q = self.fc1(q)
        q = self.fc2(q)

        return q

################# CONVERT THE MODEL HERE : ###################
# Load the model
model = CNNIQAnet()
model.load_state_dict(torch.load('C:/Users/win 10/Desktop/CNNIQA/CNNIQA/results/CNNIQA-EuroSat-EXP0-lr=0.001'))
model.eval()  # Set the model to evaluation mode

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the TorchScript model
scripted_model.save('C:/Users/win 10/Desktop/optim_CNNIQA/torchscript_model.pt')