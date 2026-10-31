import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Resnet Architecture for face recognition 
class myResnet20(nn.Module):
    def __init__(self, embedding_dim=128):
        super(myResnet20, self).__init__()
        
        # intial conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Block1 (16 channels)
        # Stage 1
        self.b1_s1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )

        # stage 2
        self.b1_s2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )

        # stage 3
        self.b1_s3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )

        # Block2 (16 -> 32 channels, stride=2 downsamples spatial dims)
        # Stage 1
        self.b2_s1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        # 1x1 projection shortcut for dimension change
        self.b2_shortcut = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )

        # stage 2
        self.b2_s2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        # stage 3
        self.b2_s3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        # Block3 (32 -> 64 channels, stride=2 downsamples spatial dims)
        # Stage 1
        self.b3_s1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # 1x1 projection shortcut for dimension change
        self.b3_shortcut = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )

        # stage 2
        self.b3_s2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # stage 3
        self.b3_s3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # Block4 (64 -> 128 channels, stride=2 downsamples spatial dims)
        # Stage 1
        self.b4_s1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # 1x1 projection shortcut for dimension change
        self.b4_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        # stage 2
        self.b4_s2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # stage 3
        self.b4_s3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))

        # Block1 Stage1 (identity shortcut)
        identity = out
        out = self.b1_s1(out)
        out = F.relu(out + identity)

        # Block1 Stage2
        identity = out
        out = self.b1_s2(out)
        out = F.relu(out + identity)

        # Block1 Stage3
        identity = out
        out = self.b1_s3(out)
        out = F.relu(out + identity)

        # Block2 Stage1 (projection shortcut: 16->32, stride 2)
        identity = self.b2_shortcut(out)
        out = self.b2_s1(out)
        out = F.relu(out + identity)

        # Block2 Stage2
        identity = out
        out = self.b2_s2(out)
        out = F.relu(out + identity)

        # Block2 Stage3
        identity = out
        out = self.b2_s3(out)
        out = F.relu(out + identity)

        # Block3 Stage1 (projection shortcut: 32->64, stride 2)
        identity = self.b3_shortcut(out)
        out = self.b3_s1(out)
        out = F.relu(out + identity)

        # Block3 Stage2
        identity = out
        out = self.b3_s2(out)
        out = F.relu(out + identity)

        # Block3 Stage3
        identity = out
        out = self.b3_s3(out)
        out = F.relu(out + identity)

        # Block4 Stage1 (projection shortcut: 64->128, stride 2)
        identity = self.b4_shortcut(out)
        out = self.b4_s1(out)
        out = F.relu(out + identity)

        # Block4 Stage2
        identity = out
        out = self.b4_s2(out)
        out = F.relu(out + identity)

        # Block4 Stage3
        identity = out
        out = self.b4_s3(out)
        out = F.relu(out + identity)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=1) # normalize for face embedding
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_face_resnet(embedding_dim=128, device=device):
    torch.manual_seed(42)
    model = myResnet20(embedding_dim=embedding_dim).to(device)
    model.eval()
    return model

FaceResNet = myResnet20

if __name__ == '__main__':
    model = myResnet20().to(device)
    model.eval()
    summary(model, input_size=(1, 3, 112, 112))
