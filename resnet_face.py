import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Resnet Architecture for face recognition (Lab 10 style)
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

        # spatial pooling across stages to capture face geometry
        self.pool_b2 = nn.AdaptiveAvgPool2d((4, 4)) # 32*16 = 512
        self.pool_b3 = nn.AdaptiveAvgPool2d((2, 2)) # 64*4 = 256
        self.fc = nn.Linear(512 + 256, embedding_dim)

    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))

        # Block1
        out = F.relu(self.b1_s1(out) + out)
        out = F.relu(self.b1_s2(out) + out)
        out = F.relu(self.b1_s3(out) + out)

        # Block2
        id2 = self.b2_shortcut(out)
        out = F.relu(self.b2_s1(out) + id2)
        out = F.relu(self.b2_s2(out) + out)
        out = F.relu(self.b2_s3(out) + out)
        f_b2 = self.pool_b2(out)

        # Block3
        id3 = self.b3_shortcut(out)
        out = F.relu(self.b3_s1(out) + id3)
        out = F.relu(self.b3_s2(out) + out)
        out = F.relu(self.b3_s3(out) + out)
        f_b3 = self.pool_b3(out)

        feat = torch.cat([torch.flatten(f_b2, 1), torch.flatten(f_b3, 1)], dim=1)
        out = self.fc(feat)
        out = F.normalize(out, p=2, dim=1)
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
