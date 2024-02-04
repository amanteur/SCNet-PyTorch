import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=input_dim)
        self.rnn = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """
        Input: (B, T, D)
        Output: (B, T, D)
        """
        x = x.transpose(1, 2)
        x = self.groupnorm(x)
        x = x.transpose(1, 2)

        x, (hidden, _) = self.rnn(x)
        x = self.fc(x)
        return x


class RFFTModule(nn.Module):
    def __init__(self, inverse: bool = False):
        super().__init__()
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, F, T, D)
        Output: (B, F, T // 2 + 1, D * 2)
        """
        B, F, T, D = x.shape
        if not self.inverse:
            x = torch.fft.rfft(x, dim=2)
            x = torch.view_as_real(x)
            x = x.reshape(B, F, T // 2 + 1, D * 2)
        else:
            x = x.reshape(B, F, T, D // 2, 2)
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, dim=2)
        return x

    def extra_repr(self) -> str:
        return f"inverse={self.inverse}"


class DualPathRNN(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(1, n_layers + 1):
            if i % 2 == 1:
                layer = nn.ModuleList(
                    [
                        RNNModule(input_dim=input_dim, hidden_dim=hidden_dim),
                        RNNModule(input_dim=input_dim, hidden_dim=hidden_dim),
                        RFFTModule(inverse=False),
                    ]
                )
            else:
                layer = nn.ModuleList(
                    [
                        RNNModule(input_dim=input_dim * 2, hidden_dim=hidden_dim * 2),
                        RNNModule(input_dim=input_dim * 2, hidden_dim=hidden_dim * 2),
                        RFFTModule(inverse=True),
                    ]
                )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, F, T, D)
        Output: (B, F, T, D)
        """
        for time_layer, freq_layer, rfft_layer in self.layers:
            B, F, T, D = x.shape

            x = x.reshape((B * F), T, D)
            x = time_layer(x)
            x = x.reshape(B, F, T, D)
            x = x.permute(0, 2, 1, 3)

            x = x.reshape((B * T), F, D)
            x = freq_layer(x)
            x = x.reshape(B, T, F, D)
            x = x.permute(0, 2, 1, 3)

            x = rfft_layer(x)

        return x
