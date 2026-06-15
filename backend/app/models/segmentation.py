import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3,
            padding=dilation, dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int, dilations: list):
        super().__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(dilations[i % len(dilations)], num_f_maps, num_f_maps)
             for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageTCN(nn.Module):
    def __init__(
        self,
        num_stages: int,
        num_layers: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
        dilations: list = None,
        random_seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.num_stages = num_stages
        self.num_classes = num_classes

        self.stage1 = SingleStageTCN(num_layers, num_f_maps, dim, num_classes, dilations)
        self.stages = nn.ModuleList(
            [SingleStageTCN(num_layers, num_f_maps, num_classes, num_classes, dilations)
             for _ in range(num_stages - 1)]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list:
        x = x.transpose(1, 2)
        outputs = []
        out = self.stage1(x)
        outputs.append(out.transpose(1, 2))

        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out.transpose(1, 2))

        return outputs

    @torch.no_grad()
    def predict(self, features: torch.Tensor, device: str = "cpu") -> tuple:
        self.eval()
        self.to(device)
        features = features.to(device).unsqueeze(0)

        outputs = self.forward(features)
        final_logits = outputs[-1].squeeze(0)
        probs = F.softmax(final_logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)

        return pred_labels.cpu().numpy(), probs.cpu().numpy()
