"""
RNN-T Loss Implementation using torchaudio's implementation
Reference: https://pytorch.org/audio/stable/generated/torchaudio.functional.rnnt_loss.html

The RNN-T loss computes the negative log-likelihood of all possible alignments
between input and target sequences using the forward-backward algorithm.
"""

import torch
import torch.nn as nn
try:
    from torchaudio.functional import rnnt_loss
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not available. Using custom RNN-T loss implementation.")


class RNNTLoss(nn.Module):
    """
    RNN-T Loss wrapper

    Args:
        blank: Index of blank token (default: 0)
        reduction: Reduction method ('mean', 'sum', 'none')
        clamp: Clamp probabilities to avoid log(0)
    """
    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        clamp: float = 1e-7
    ):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.clamp = clamp

        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for RNN-T loss. "
                "Install with: pip install torchaudio"
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RNN-T loss

        Args:
            logits: Joint network output (batch, max_time, max_target_length+1, vocab_size)
            targets: Target labels (batch, max_target_length)
            input_lengths: Actual input sequence lengths (batch,)
            target_lengths: Actual target sequence lengths (batch,)

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', else (batch,))
        """
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Clamp to avoid log(0)
        log_probs = torch.clamp(log_probs, min=torch.log(torch.tensor(self.clamp, device=log_probs.device)))

        # Compute RNN-T loss
        loss = rnnt_loss(
            logits=log_probs,
            targets=targets.int(),
            logit_lengths=input_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.blank,
            reduction=self.reduction
        )

        return loss


class CustomRNNTLoss(nn.Module):
    """
    Custom RNN-T Loss implementation using dynamic programming
    This is a simplified version for educational purposes.

    For production use, please use torchaudio's optimized implementation.
    """
    def __init__(self, blank: int = 0):
        super().__init__()
        self.blank = blank

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified RNN-T loss using forward algorithm

        Note: This is not optimized and should only be used for understanding.
        """
        batch_size = logits.size(0)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        losses = []

        for b in range(batch_size):
            T = input_lengths[b].item()
            U = target_lengths[b].item()

            # Forward variables: alpha[t, u] = log P(y_1:u, t)
            alpha = torch.full(
                (T + 1, U + 1),
                float('-inf'),
                device=logits.device
            )

            # Initialize: alpha[0, 0] = log(1) = 0
            alpha[0, 0] = 0.0

            # Target sequence for this batch
            target_seq = targets[b, :U]

            # Forward pass
            for t in range(T):
                for u in range(U + 1):
                    if alpha[t, u] == float('-inf'):
                        continue

                    # Emit blank (stay at same label position)
                    blank_prob = log_probs[b, t, u, self.blank]
                    alpha[t + 1, u] = torch.logaddexp(
                        alpha[t + 1, u],
                        alpha[t, u] + blank_prob
                    )

                    # Emit label (advance label position)
                    if u < U:
                        label = target_seq[u].item()
                        label_prob = log_probs[b, t, u, label]
                        alpha[t, u + 1] = torch.logaddexp(
                            alpha[t, u + 1],
                            alpha[t, u] + label_prob
                        )

            # Final probability
            loss = -alpha[T, U]
            losses.append(loss)

        return torch.stack(losses).mean()


def test_loss():
    """Test RNN-T loss computation"""
    batch_size = 2
    max_time = 10
    max_label_length = 5
    vocab_size = 10

    # Create dummy data
    logits = torch.randn(batch_size, max_time, max_label_length + 1, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, max_label_length))
    input_lengths = torch.tensor([10, 8])
    target_lengths = torch.tensor([5, 4])

    # Test with torchaudio loss
    if TORCHAUDIO_AVAILABLE:
        criterion = RNNTLoss(blank=0, reduction='mean')
        loss = criterion(logits, targets, input_lengths, target_lengths)
        print(f"RNN-T Loss (torchaudio): {loss.item():.4f}")

    # Test with custom loss
    custom_criterion = CustomRNNTLoss(blank=0)
    custom_loss = custom_criterion(logits, targets, input_lengths, target_lengths)
    print(f"RNN-T Loss (custom): {custom_loss.item():.4f}")


if __name__ == "__main__":
    test_loss()
