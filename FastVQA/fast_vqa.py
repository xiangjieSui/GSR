import torch.nn as nn
from FastVQA.conv_backbone import convnext_3d_tiny
from FastVQA.swin_backbone import SwinTransformer3D as VideoBackbone
import torch
import matplotlib.pyplot as plt
from FastVQA.xclip_backbone import build_x_clip_model


class EnsembleHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
        scanpath.shape = (B, N, T * 2) >> weight.shape = (B, N)
    """

    def __init__(
        self, in_channels=30, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(in_channels, hidden_channels)
        self.rnn = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.fc_last = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x, _ = self.rnn(self.fc_hid(x))
        weight = self.fc_last(self.dropout(x))
        weight = torch.softmax(weight, dim=1).squeeze()
        return weight


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(
            self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score


class DiViDeAddEvaluator(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys='fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4, 4, 4)},
                      fragments={"window_size": (4, 4, 4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):

        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        for key, hypers in backbone.items():
            # print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            # print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b)
        if divide_head:
            # print(divide_head)
            for key in backbone:
                if key not in self.backbone_preserve_keys:
                    continue
                b = VQAHead(**vqa_head)
                # print("Setting head:", "vqa_head")
                setattr(self, "vqa_head", b)
        else:
            self.vqa_head = VQAHead(**vqa_head)


    def forward(self, vclips, masking, inference=True, output_score_map=False, **kwargs):
        if inference:
            self.eval()
            self.device = vclips.device
            with torch.no_grad():
                feat = getattr(self, "fragments_backbone")(
                    vclips, multi=self.multi, layer=self.layer, **kwargs)
                scores = getattr(self, "vqa_head")(feat)

            self.train()

            new_scores = []
            for i in range(scores.shape[0]):
                split = int((masking[i] + 1) / 2)
                current_score = scores[i][0][:split]
                if output_score_map:
                    for j in range(split):
                        plt.imshow(current_score.cpu()[j], cmap='Blues',
                                    interpolation='nearest')
                        if j == 0:
                            plt.colorbar()
                        plt.savefig('heatmap_' + str(j) + '.png')
                new_scores += [torch.sum(current_score) / split]

            scores = torch.stack(new_scores).view(-1, 1)

            return scores

        else:
            self.train()
            self.device = vclips.device
            feats = getattr(self, "fragments_backbone")(
                vclips, multi=self.multi, layer=self.layer, **kwargs)
            scores = getattr(self, "vqa_head")(feats)

            new_scores = []
            for i in range(scores.shape[0]):
                split = int((masking[i] + 1) / 2)
                current_score = scores[i][0][:split]
                new_scores += [torch.sum(current_score) / split]
            scores = torch.stack(new_scores).view(-1, 1)

            return scores
