"""
RNN-T Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
import os

from model import RNNT
from loss import RNNTLoss


class DummyASRDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    Replace this with your actual speech recognition dataset.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 80,
        vocab_size: int = 30,
        max_time: int = 200,
        max_label_length: int = 50
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.max_time = max_time
        self.max_label_length = max_label_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random audio features (mel-spectrogram)
        time_length = np.random.randint(50, self.max_time)
        inputs = torch.randn(time_length, self.input_dim)

        # Generate random target sequence
        label_length = np.random.randint(5, min(self.max_label_length, time_length // 2))
        targets = torch.randint(1, self.vocab_size, (label_length,))

        return inputs, targets


def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads sequences to same length within batch
    """
    inputs, targets = zip(*batch)

    # Get lengths
    input_lengths = torch.tensor([x.size(0) for x in inputs])
    target_lengths = torch.tensor([y.size(0) for y in targets])

    # Pad inputs
    max_input_len = input_lengths.max().item()
    input_dim = inputs[0].size(1)
    padded_inputs = torch.zeros(len(inputs), max_input_len, input_dim)
    for i, inp in enumerate(inputs):
        padded_inputs[i, :inp.size(0), :] = inp

    # Pad targets
    max_target_len = target_lengths.max().item()
    padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
    for i, tgt in enumerate(targets):
        padded_targets[i, :tgt.size(0)] = tgt

    return padded_inputs, input_lengths, padded_targets, target_lengths


class Trainer:
    """
    RNN-T Trainer
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss and optimizer
        self.criterion = RNNTLoss(blank=0, reduction='mean')
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, input_lengths, targets, target_lengths) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            input_lengths = input_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass
            logits = self.model(inputs, input_lengths, targets, target_lengths)

            # Compute loss
            loss = self.criterion(logits, targets, input_lengths, target_lengths)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model"""
        if self.val_loader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0

        for inputs, input_lengths, targets, target_lengths in tqdm(
            self.val_loader, desc='Validation'
        ):
            # Move to device
            inputs = inputs.to(self.device)
            input_lengths = input_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass
            logits = self.model(inputs, input_lengths, targets, target_lengths)

            # Compute loss
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }

        # Save latest checkpoint
        path = os.path.join(self.checkpoint_dir, 'last.pt')
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, path)
            print(f'Saved best model with val_loss: {val_loss:.4f}')

    def train(self, num_epochs: int):
        """Train for multiple epochs"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}')

            # Validate
            val_loss = self.validate()
            if val_loss != float('inf'):
                print(f'Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}')

                # Update scheduler
                self.scheduler.step(val_loss)

                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best)
            else:
                self.save_checkpoint(epoch, train_loss, False)

            print('-' * 80)


def main():
    # Hyperparameters
    INPUT_DIM = 80  # Mel-spectrogram features
    VOCAB_SIZE = 30  # Example vocabulary size (including blank)
    ENCODER_HIDDEN = 256
    PREDICTOR_HIDDEN = 256
    JOINT_HIDDEN = 256
    EMBEDDING_DIM = 128

    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Create datasets
    train_dataset = DummyASRDataset(
        num_samples=1000,
        input_dim=INPUT_DIM,
        vocab_size=VOCAB_SIZE
    )
    val_dataset = DummyASRDataset(
        num_samples=200,
        input_dim=INPUT_DIM,
        vocab_size=VOCAB_SIZE
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = RNNT(
        input_dim=INPUT_DIM,
        vocab_size=VOCAB_SIZE,
        encoder_hidden=ENCODER_HIDDEN,
        predictor_hidden=PREDICTOR_HIDDEN,
        joint_hidden=JOINT_HIDDEN,
        embedding_dim=EMBEDDING_DIM,
        num_encoder_layers=2,
        num_predictor_layers=2,
        dropout=0.1
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        checkpoint_dir='./checkpoints'
    )

    # Train
    trainer.train(num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
