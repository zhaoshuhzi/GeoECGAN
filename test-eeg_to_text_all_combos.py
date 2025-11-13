"""
synthetic_eeg_to_text_test.py

功能：
1. 生成模拟 EEG 数据 (随机) 和模拟文本 token 序列 (随机)。
2. 定义 3 种 Encoder（几何约束 / NPI / 几何+NPI）。
3. 定义 3 种 Decoder（GAN / BERT / Diffusion）。
4. 9 种 Encoder-Decoder 组合全部跑一遍，在模拟数据上训练 1 个 epoch，确认能跑通。

依赖：
    pip install torch

运行：
    python synthetic_eeg_to_text_test.py
"""

import itertools
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===========================
# 枚举类型
# ===========================

EncoderType = Literal["geometry", "npi", "geometry_npi"]
DecoderType = Literal["gan", "bert", "diffusion"]

ENCODER_CANDIDATES: Tuple[EncoderType, ...] = ("geometry", "npi", "geometry_npi")
DECODER_CANDIDATES: Tuple[DecoderType, ...] = ("gan", "bert", "diffusion")


# ===========================
# 全局配置
# ===========================

@dataclass
class EEGTextConfig:
    eeg_channels: int = 32          # EEG 通道数（模拟）
    eeg_timepoints: int = 128       # 单 trial 时间点（模拟）
    latent_dim: int = 64           # 编码器输出潜在维度
    vocab_size: int = 1000         # 词表大小（模拟）
    max_seq_len: int = 16          # 文本最大长度（模拟）
    batch_size: int = 8            # 批大小
    num_samples: int = 64          # 模拟数据的样本数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================
# 模拟数据集
# ===========================

class SyntheticEEGTextDataset(Dataset):
    """
    简单模拟数据集：
    - EEG: 正态分布随机数 [num_samples, C, T]
    - Text tokens: [num_samples, L]，每个位置是 [0, vocab_size-1] 的整数
    """

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.cfg = cfg
        self.eeg_data = torch.randn(cfg.num_samples, cfg.eeg_channels, cfg.eeg_timepoints)
        self.text_tokens = torch.randint(
            low  = 0,
            high = cfg.vocab_size,
            size = (cfg.num_samples, cfg.max_seq_len),
        )

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]          # [C, T]
        tokens = self.text_tokens[idx]    # [L]
        return eeg, tokens


# ===========================
# 编码器定义（骨架）
# ===========================

class GeometryEncoder(nn.Module):
    """几何约束 Encoder（占位：可替换为 eigenmodes / Laplace-Beltrami 等）"""

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.eeg_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 128, 1]
        )
        self.fc = nn.Linear(128, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.conv(x).squeeze(-1)  # [B, 128]
        z = self.fc(h)                # [B, latent_dim]
        return z


class NPIEncoder(nn.Module):
    """NPI Encoder（占位：这里用 BiLSTM 模拟“信息流传播”特征）"""

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.cfg = cfg
        self.rnn = nn.LSTM(
            input_size=cfg.eeg_channels,
            hidden_size=cfg.latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(cfg.latent_dim * 2, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        x_seq = x.permute(0, 2, 1)
        _, (hn, _) = self.rnn(x_seq)   # hn: [num_layers*2, B, H]
        h_last = torch.cat([hn[-2], hn[-1]], dim=-1)  # [B, 2H]
        z = self.fc(h_last)                         # [B, latent_dim]
        return z


class GeometryNPIEncoder(nn.Module):
    """几何约束 + NPI 联合 Encoder（并行分支 + MLP 融合）"""

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.geo_encoder = GeometryEncoder(cfg)
        self.npi_encoder = NPIEncoder(cfg)
        self.fc_fuse = nn.Linear(cfg.latent_dim * 2, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_geo = self.geo_encoder(x)
        z_npi = self.npi_encoder(x)
        z = torch.cat([z_geo, z_npi], dim=-1)
        z = self.fc_fuse(z)
        return z


def build_encoder(encoder_type: EncoderType, cfg: EEGTextConfig) -> nn.Module:
    if encoder_type == "geometry":
        return GeometryEncoder(cfg)
    elif encoder_type == "npi":
        return NPIEncoder(cfg)
    elif encoder_type == "geometry_npi":
        return GeometryNPIEncoder(cfg)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# ===========================
# 解码器定义（骨架）
# return logits: [B, L, vocab_size]
# ===========================

class GANTextDecoder(nn.Module):
    """简化版 GAN-style 文本解码器（仅生成器骨架）"""

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_to_seq = nn.Linear(cfg.latent_dim, cfg.max_seq_len * cfg.latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=4,
            dim_feedforward=cfg.latent_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(cfg.latent_dim, cfg.vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.latent_to_seq(z).view(B, self.cfg.max_seq_len, self.cfg.latent_dim)
        h = self.transformer(h)
        logits = self.fc_out(h)
        return logits


class BERTTextDecoder(nn.Module):
    """简化版 BERT-like 文本解码器（Encoder-only block 做条件生成）"""

    def __init__(self, cfg: EEGTextConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=8,
            dim_feedforward=cfg.latent_dim * 4,
            batch_first=True,
        )
        self.bert_block = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc_out = nn.Linear(cfg.latent_dim, cfg.vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.latent_to_hidden(z).unsqueeze(1)                  # [B, 1, D]
        h = h.repeat(1, self.cfg.max_seq_len, 1)                    # [B, L, D]
        positions = torch.arange(self.cfg.max_seq_len, device=z.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0)              # [1, L, D]
        h = h + pos_emb
        h = self.bert_block(h)
        logits = self.fc_out(h)
        return logits


class DiffusionTextDecoder(nn.Module):
    """极简版 Diffusion-style 文本解码器骨架（toy 去噪网络）"""

    def __init__(self, cfg: EEGTextConfig, num_steps: int = 3):
        super().__init__()
        self.cfg = cfg
        self.num_steps = num_steps
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.max_seq_len * cfg.latent_dim)
        self.denoise_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.fc_out = nn.Linear(cfg.latent_dim, cfg.vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.latent_to_hidden(z).view(B, self.cfg.max_seq_len, self.cfg.latent_dim)
        # 加一点噪声
        h = h + torch.randn_like(h) * 0.1
        # 迭代“去噪”
        for _ in range(self.num_steps):
            h = h + self.denoise_net(h)
        logits = self.fc_out(h)
        return logits


def build_decoder(decoder_type: DecoderType, cfg: EEGTextConfig) -> nn.Module:
    if decoder_type == "gan":
        return GANTextDecoder(cfg)
    elif decoder_type == "bert":
        return BERTTextDecoder(cfg)
    elif decoder_type == "diffusion":
        return DiffusionTextDecoder(cfg)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")


# ===========================
# 整体 EEG-to-Text 模型
# ===========================

class EEGToTextModel(nn.Module):
    def __init__(self, encoder_type: EncoderType, decoder_type: DecoderType, cfg: EEGTextConfig):
        super().__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder = build_encoder(encoder_type, cfg)
        self.decoder = build_decoder(decoder_type, cfg)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, C, T]
        return: logits [B, L, vocab_size]
        """
        z = self.encoder(eeg)
        logits = self.decoder(z)
        return logits


# ===========================
# 训练函数（用模拟数据）
# ===========================

def train_one_epoch(
    model: EEGToTextModel,
    cfg: EEGTextConfig,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    device = cfg.device
    total_loss = 0.0
    num_batches = 0

    for eeg, tokens in dataloader:
        eeg = eeg.to(device)           # [B, C, T]
        tokens = tokens.to(device)     # [B, L]

        logits = model(eeg)            # [B, L, vocab]
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            tokens.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


# ===========================
# 主函数：9 种组合在模拟数据上跑通
# ===========================

def main():
    torch.manual_seed(42)
    cfg = EEGTextConfig()
    print(f"Using device: {cfg.device}")

    # 构建模拟数据集 & DataLoader
    dataset = SyntheticEEGTextDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    epochs_per_combo = 1  # 每种组合只跑 1 个 epoch，验证能跑通即可

    # 9 种组合：笛卡尔积
    all_combos = list(itertools.product(ENCODER_CANDIDATES, DECODER_CANDIDATES))

    for idx, (enc_type, dec_type) in enumerate(all_combos, start=1):
        print("=" * 80)
        print(f"[Combo {idx}/9] Encoder: {enc_type:12s} | Decoder: {dec_type:9s}")
        print("=" * 80)

        model = EEGToTextModel(enc_type, dec_type, cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, epochs_per_combo + 1):
            loss = train_one_epoch(model, cfg, dataloader, optimizer)
            print(f"  Epoch {epoch}: mean loss = {loss:.4f}")

        # 做一次前向检查输出形状
        model.eval()
        with torch.no_grad():
            eeg, tokens = next(iter(dataloader))
            eeg = eeg.to(cfg.device)
            logits = model(eeg)
            print(f"  Forward check - logits shape: {tuple(logits.shape)}  # [B, L, vocab_size]")

    print("\n所有 9 种编码器-解码器组合在模拟数据上已成功跑通。")


if __name__ == "__main__":
    main()
