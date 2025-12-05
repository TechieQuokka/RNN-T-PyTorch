"""
RNN-T (Recurrent Neural Network Transducer) Implementation
Based on "Sequence Transduction with Recurrent Neural Networks" by Alex Graves (2012)
https://arxiv.org/abs/1211.3711
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Transcription Network: Encodes input acoustic features into hidden representations

    Args:
        input_dim: Dimension of input features (e.g., 80 for mel-spectrogram)
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, input_dim)
            lengths: Actual lengths of sequences (batch,)

        Returns:
            Encoded features of shape (batch, time, hidden_dim)
        """
        if lengths is not None:
            # Pack padded sequence for efficient computation
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        x, _ = self.lstm(x)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.layer_norm(x)
        return x


class PredictionNetwork(nn.Module):
    """
    Prediction Network: Language model that predicts next token based on previous outputs

    Args:
        vocab_size: Size of output vocabulary (including blank)
        embedding_dim: Dimension of token embeddings
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        y: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            y: Previous output tokens of shape (batch, label_length)
            lengths: Actual lengths of label sequences (batch,)
            state: Previous LSTM state (h, c)

        Returns:
            Prediction network output of shape (batch, label_length, hidden_dim)
            New LSTM state (h, c)
        """
        x = self.embedding(y)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        x, state = self.lstm(x, state)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.layer_norm(x)
        return x, state


class JointNetwork(nn.Module):
    """
    Joint Network: Combines encoder and prediction network outputs

    Args:
        input_dim: Input dimension (should be encoder_dim + predictor_dim)
        hidden_dim: Hidden dimension of feedforward network
        vocab_size: Size of output vocabulary (including blank)
    """
    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        vocab_size: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(encoder_dim + predictor_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        predictor_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: Encoder output (batch, time, encoder_dim)
            predictor_out: Predictor output (batch, label_length, predictor_dim)

        Returns:
            Joint network output (batch, time, label_length, vocab_size)
        """
        # Expand dimensions for broadcasting
        # encoder_out: (B, T, 1, encoder_dim)
        # predictor_out: (B, 1, U, predictor_dim)
        encoder_out = encoder_out.unsqueeze(2)
        predictor_out = predictor_out.unsqueeze(1)

        # Concatenate and pass through network
        # (B, T, U, encoder_dim + predictor_dim)
        concat = torch.cat([
            encoder_out.expand(-1, -1, predictor_out.size(2), -1),
            predictor_out.expand(-1, encoder_out.size(1), -1, -1)
        ], dim=-1)

        out = self.fc1(concat)
        out = self.tanh(out)
        out = self.fc2(out)  # (B, T, U, vocab_size)

        return out


class RNNT(nn.Module):
    """
    Complete RNN-T Model

    Args:
        input_dim: Dimension of input acoustic features
        vocab_size: Size of output vocabulary (including blank token at index 0)
        encoder_hidden: Hidden dimension of encoder
        predictor_hidden: Hidden dimension of prediction network
        joint_hidden: Hidden dimension of joint network
        num_encoder_layers: Number of encoder LSTM layers
        num_predictor_layers: Number of predictor LSTM layers
        dropout: Dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        encoder_hidden: int = 512,
        predictor_hidden: int = 512,
        joint_hidden: int = 512,
        embedding_dim: int = 256,
        num_encoder_layers: int = 2,
        num_predictor_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.blank_idx = 0  # Blank token index

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        self.predictor = PredictionNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=predictor_hidden,
            num_layers=num_predictor_layers,
            dropout=dropout
        )

        self.joint = JointNetwork(
            encoder_dim=encoder_hidden,
            predictor_dim=predictor_hidden,
            hidden_dim=joint_hidden,
            vocab_size=vocab_size
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training

        Args:
            inputs: Input acoustic features (batch, max_time, input_dim)
            input_lengths: Actual input lengths (batch,)
            targets: Target labels (batch, max_label_length)
            target_lengths: Actual target lengths (batch,)

        Returns:
            Joint network logits (batch, max_time, max_label_length+1, vocab_size)
        """
        # Encode inputs
        encoder_out = self.encoder(inputs, input_lengths)

        # Prepend blank token to targets for prediction network
        batch_size = targets.size(0)
        blank_prefix = torch.zeros(
            batch_size, 1,
            dtype=targets.dtype,
            device=targets.device
        )
        targets_with_blank = torch.cat([blank_prefix, targets], dim=1)

        # Prediction network forward
        predictor_out, _ = self.predictor(
            targets_with_blank,
            target_lengths + 1  # +1 for blank prefix
        )

        # Joint network
        logits = self.joint(encoder_out, predictor_out)

        return logits

    def greedy_decode(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_symbols: int = 100
    ) -> list:
        """
        Greedy decoding for inference

        Args:
            inputs: Input acoustic features (batch, time, input_dim)
            input_lengths: Actual input lengths (batch,)
            max_symbols: Maximum number of symbols to output

        Returns:
            List of decoded sequences (list of lists)
        """
        self.eval()
        with torch.no_grad():
            # Encode inputs
            encoder_out = self.encoder(inputs, input_lengths)
            batch_size, max_time, _ = encoder_out.shape

            decoded_sequences = []

            for b in range(batch_size):
                # Get sequence length for this batch item
                seq_len = input_lengths[b].item()

                # Initialize with blank token
                current_token = torch.tensor(
                    [[self.blank_idx]],
                    dtype=torch.long,
                    device=inputs.device
                )

                # Initial predictor state
                predictor_state = None

                decoded = []

                for t in range(seq_len):
                    # Get encoder output at time t
                    enc_t = encoder_out[b:b+1, t:t+1, :]  # (1, 1, encoder_dim)

                    # Keep predicting until blank
                    for _ in range(max_symbols):
                        # Get predictor output
                        pred_out, predictor_state = self.predictor(
                            current_token,
                            state=predictor_state
                        )

                        # Joint network
                        logits = self.joint(enc_t, pred_out)  # (1, 1, 1, vocab_size)
                        logits = logits.squeeze(0).squeeze(0).squeeze(0)  # (vocab_size,)

                        # Greedy selection
                        pred = torch.argmax(logits, dim=-1).item()

                        if pred == self.blank_idx:
                            # Move to next time step
                            break
                        else:
                            decoded.append(pred)
                            current_token = torch.tensor(
                                [[pred]],
                                dtype=torch.long,
                                device=inputs.device
                            )

                decoded_sequences.append(decoded)

            return decoded_sequences


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    input_dim = 80  # mel-spectrogram features
    vocab_size = 30  # example vocab size (including blank)
    max_time = 100
    max_label_length = 20

    # Create model
    model = RNNT(
        input_dim=input_dim,
        vocab_size=vocab_size,
        encoder_hidden=256,
        predictor_hidden=256,
        joint_hidden=256,
        embedding_dim=128
    )

    # Create dummy data
    inputs = torch.randn(batch_size, max_time, input_dim)
    input_lengths = torch.tensor([100, 95, 90, 85])
    targets = torch.randint(1, vocab_size, (batch_size, max_label_length))
    target_lengths = torch.tensor([20, 18, 15, 12])

    # Forward pass
    logits = model(inputs, input_lengths, targets, target_lengths)
    print(f"Logits shape: {logits.shape}")  # (batch, time, label_length+1, vocab_size)

    # Greedy decoding
    decoded = model.greedy_decode(inputs, input_lengths)
    print(f"Decoded sequences: {decoded}")

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
