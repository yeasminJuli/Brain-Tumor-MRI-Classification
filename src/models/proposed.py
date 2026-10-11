import torch
import torch.nn as nn
NUM_CLASSES = 4

class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers; uses 1x1 conv for skip if channels differ.
    Optionally includes a MaxPool inside the block to reduce spatial size.
    """
    def __init__(self, in_channels, out_channels, use_pool=False):
        super().__init__()
        self.use_pool = use_pool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # skip conv if channel dims change
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.relu(out)
        out = self.pool(out)
        return out

class CNNFeatureExtractor(nn.Module):
    """
    5 residual blocks:
      Block1: 3ch -> 32 (pool)
      Block2: 32 -> 64 (pool)
      Block3: 64 -> 128 (pool)
      Block4: 128 -> 128 (pool)
      Block5: 128 -> 256 (no pool here)
    Final adaptive pooling to (8,8)
    """
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(3, 32, use_pool=True)
        self.block2 = ResidualBlock(32, 64, use_pool=True)
        self.block3 = ResidualBlock(64, 128, use_pool=True)
        self.block4 = ResidualBlock(128, 128, use_pool=True)
        self.block5 = ResidualBlock(128, 256, use_pool=False)
        self.final_pool = nn.AdaptiveAvgPool2d((8, 8))  # final featuremap 256 x 8 x 8

    def forward(self, x):
        x = self.block1(x)   # spatial reduced by /2
        x = self.block2(x)   # /4
        x = self.block3(x)   # /8
        x = self.block4(x)   # /16
        x = self.block5(x)   # remains /16
        x = self.final_pool(x)
        return x

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 128)
        self.tanh = nn.Tanh()
        self.score = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden*2)
        e = self.tanh(self.attn(lstm_outputs))      # (batch, seq_len, 128)
        scores = self.score(e)                     # (batch, seq_len, 1)
        alpha = self.softmax(scores)               # (batch, seq_len, 1)
        context = torch.sum(alpha * lstm_outputs, dim=1)  # (batch, hidden*2)
        return context, alpha

class HybridCNNBiLSTMAttention(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers
        # After CNN final_pool -> (batch, 256, 8, 8)
        c_channels = 256
        c_w = 8
        self.lstm_input_size = c_channels * c_w  # 256*8 = 2048
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(self.hidden_size)
        # FC layers after attention context vector (hidden*2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x -> CNN
        x = self.cnn(x)                        # (batch, 256, 8, 8)
        b, c, h, w = x.size()
        # Flatten along height to create sequence length = h
        # sequence_length = h (8), input_size = c * w (256*8)
        x = x.view(b, h, c * w)               # (batch, seq_len=h, input_size)
        lstm_out, _ = self.bilstm(x)          # (batch, seq_len, hidden*2)
        context, attn_weights = self.attention(lstm_out)  # context: (batch, hidden*2)
        out = self.fc(context)                # (batch, num_classes)
        return out, attn_weights
