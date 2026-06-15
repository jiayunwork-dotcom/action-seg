import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowFastSlowBranch(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.layer1 = self._make_layer(64, 128, 1)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        self.layer4 = self._make_layer(512, feature_dim, 2)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def _make_layer(self, in_channels: int, out_channels: int, stride_t: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 3, 3),
                stride=(stride_t, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1)
        return x


class FeatureExtractor3D(nn.Module):
    def __init__(self, feature_dim: int = 512, random_seed: int = 42):
        super().__init__()
        torch.manual_seed(random_seed)
        self.backbone = SlowFastSlowBranch(feature_dim=feature_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def extract(self, frames: torch.Tensor, batch_size: int = 2, device: str = "cpu") -> torch.Tensor:
        self.eval()
        self.to(device)

        all_features = []
        total_clips = frames.shape[0]

        for start in range(0, total_clips, batch_size):
            end = min(start + batch_size, total_clips)
            batch = frames[start:end].to(device)
            feats = self.forward(batch)
            all_features.append(feats.cpu())

        return torch.cat(all_features, dim=0)
