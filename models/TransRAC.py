"""TransRAC network"""
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from models.cnn import DepthwiseSeparableConv
import random
import copy

class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        # if attn_mask:
        #     scores = scores.masked_fill_(attn_mask, -np.inf)
        # scores = self.softmax(scores)
        # scores = self.dropout(scores)  # [B,head, F, F]
        scores2 = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B,Head, F, F]
        scores2 = self.softmax(scores2)

        # q, k = map(lambda feat: feat / feat.shape[-1] ** .5, [q, k])
        scores1 = torch.matmul(q, k.transpose(-2, -1))  # [B,Head, F, F]
        scores1 = F.softmax(scores1, 2) * F.softmax(scores1, 3)

        scores = scores2 + scores1
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores, scores1, scores2  # [B,head,F, F]


class Similarity_matrix(nn.Module):
    ''' buliding similarity matrix by self-attention mechanism '''

    def __init__(self, num_heads=4, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 512
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] -> [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix, scores1, scores2 = self.attention(query, key, value, attn_mask)

        return matrix, scores1, scores2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class feature_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.conv5 = nn.Conv3d(1, 1, kernel_size=5, stride=(1, 1, 1), padding=(2, 2, 2), bias=False)
        self.conv_all = nn.Conv3d(1, 1, kernel_size=(3, 64, 64), stride=(3, 1, 1), bias=False)

    def forward(self, x):
        list1 = []
        x = torch.unsqueeze(x, dim=1)
        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        list1.append(x_1)
        list1.append(x_3)
        list1.append(x_5)
        x_cat = torch.cat(list1, dim=-3)
        x_all = self.conv_all(x_cat)
        out = x_all + x
        out = torch.squeeze(out, dim=1)
        return out


class TransferModel(nn.Module):
    def __init__(self, config, checkpoint, num_frames, scales, OPEN=False):
        super(TransferModel, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN

        self.backbone = self.load_model()  # load pretrain model

        self.Replication_padding1 = nn.ConstantPad3d((0, 0, 0, 0, 1, 1), 0)
        self.Replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.Replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)

        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))

        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
        # self.shift = shift()
        self.sims = Similarity_matrix()
        self.feature_fusion = feature_fusion()
        self.conv3x3 = nn.Conv2d(in_channels=15,  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.ds_conv = DepthwiseSeparableConv(512, 512, 5)
        self.ds_conv_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64)
        )
        # self.ds_conv_ln = nn.LayerNorm(64)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  # 线性投射层
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.FC = Prediction(512, 512, 256, 1)

    def load_model(self):
        # # # load  pretrained model of video swin transformer using mmaction and mmcv API
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # # # load hyperparameters by mmcv api
        load_checkpoint(model, self.checkpoint, map_location='cpu')
        backbone = model.backbone

        # # # load hyperparameters by pytorch
        # loaded_ckpt = torch.load(self.checkpoint)
        # backbone = model.backbone
        # net_dict = backbone.state_dict()
        # state_dict = {k: v for k, v in loaded_ckpt.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # backbone.load_state_dict(net_dict, strict=False)

        print('--------- backbone loaded ------------')

        return backbone

    def forward(self, x):
        # x: tensor([batch_size, channel, temporal_dim, height, width])
        with autocast():
            batch_size, c, num_frames, h, w = x.shape
            # scales = [1,4,8]
            # We currently only support 1, 4, 8 flames. If you want to add more scale, you can change the part and don't forget padding.
            multi_scales = []
            multi_score1 = []
            multi_score2 = []
            for scale in self.scales:
                if scale == 4:
                    x = self.Replication_padding2(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                elif scale == 8:
                    x = self.Replication_padding4(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                else:
                    crops = [x[:, :, i:i + 1, :, :] for i in range(0, self.num_frames)]

                slice = []
                # 预训练模型
                # feature extract with video SwinTransformer
                if not self.OPEN:
                    with torch.no_grad():
                        for crop in crops:
                            crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]
                            slice.append(crop)
                else:  # train  the feature extractor (video SwinTransformer backbone)
                    for crop in crops:
                        crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]
                        slice.append(crop)

                x_scale = torch.cat(slice, dim=2)  # -> [b,768,f,size,size]
                x_scale = F.relu(self.bn1(self.conv3D(x_scale)))  # ->[b,512,f,7,7]
                # print(x_scale.shape)
                # Encoder的第三步Temporal context
                x_scale = self.SpatialPooling(x_scale)  # ->[b,512,f,1,1]
                x_scale = x_scale.squeeze(3).squeeze(3)  # -> [b,512,f]
                x_scale = x_scale.transpose(1, 2)  # -> [b,f,512]

                # 新旧特征融合
                x_scale_hat = self.ds_conv(x_scale) # [32, 64, 64]
                x_scale_hat = self.ds_conv_fc(x_scale_hat) # [32, 64, 64]
                # x_scale_1 = self.ds_conv_ln(x_scale_hat + x_scale)
                x_scale_hat = torch.unsqueeze(x_scale_hat, dim=1)  # [32,1,64,64]

                # Patch Shift operation
                # x_scale = self.shift(x_scale)

                # -------- similarity matrix ---------
                x_sims = F.relu(self.sims(x_scale, x_scale, x_scale)[0])  # -> [b,4,f,f]
                score_1 =  F.relu(self.sims(x_scale, x_scale, x_scale)[1])
                score_2 =  F.relu(self.sims(x_scale, x_scale, x_scale)[2])

                # 设置随机drop，概率为0.5，置为0
                # 设置随机drop，概率为0.3，置为0
                if self.training and torch.rand(1) < 0.3: 
                    # 随机选取一个 feature map
                    drop_index = random.choice([0,1,2,3])
                    x_sims[:, drop_index, :, :] = torch.zeros_like(x_sims[:, drop_index, :, :])
                
                multi_scales.append(x_sims)
                multi_scales.append(x_scale_hat)

                multi_score1.append(score_1)
                multi_score2.append(score_2)

            x = torch.cat(multi_scales, dim=1)  # [B,4*scale_num,f,f]
            matrix_1 = torch.cat(multi_score1, dim=1)
            matrix_2 = torch.cat(multi_score2, dim=1)
            # x are the similarity matrixs
            x_matrix = x
            x = self.feature_fusion(x)
            # Period Predictor里最开始的3x3conv那里
            x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]
            x = self.dropout1(x)

            x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
            # --------- transformer encoder ------
            x = x.flatten(start_dim=2)  # ->[b,f,32*f]
            x = F.relu(self.input_projection(x))  # ->[b,f, 512]
            x = self.ln1(x)

            x = x.transpose(0, 1)  # [f,b,512]
            x = self.transEncoder(x)  # 最后Period Predictor里的Transformer
            x = x.transpose(0, 1)  # ->[b,f, 512]
            out_o = copy.deepcopy(x)
            # 最后的linear层
            x = self.FC(x)  # ->[b,f,1]
            x = x.squeeze(2)

            return x, x_matrix, out_o, matrix_1, matrix_2
