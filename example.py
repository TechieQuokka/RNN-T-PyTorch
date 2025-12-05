"""
RNN-T Example Usage and Benchmarking
"""

import torch
import time
from model import RNNT
from loss import RNNTLoss
import numpy as np


def benchmark_model():
    """
    Benchmark RNN-T model performance
    """
    print("=" * 80)
    print("RNN-T Model Benchmarking")
    print("=" * 80)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Model configurations to test
    configs = [
        {
            'name': 'Small',
            'encoder_hidden': 128,
            'predictor_hidden': 128,
            'joint_hidden': 128,
            'embedding_dim': 64,
            'num_encoder_layers': 1,
            'num_predictor_layers': 1
        },
        {
            'name': 'Medium',
            'encoder_hidden': 256,
            'predictor_hidden': 256,
            'joint_hidden': 256,
            'embedding_dim': 128,
            'num_encoder_layers': 2,
            'num_predictor_layers': 2
        },
        {
            'name': 'Large',
            'encoder_hidden': 512,
            'predictor_hidden': 512,
            'joint_hidden': 512,
            'embedding_dim': 256,
            'num_encoder_layers': 4,
            'num_predictor_layers': 2
        }
    ]

    # Test settings
    input_dim = 80
    vocab_size = 30
    batch_size = 8
    max_time = 100
    max_label_length = 20

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max time steps: {max_time}")
    print(f"  Max label length: {max_label_length}")
    print(f"  Input dim: {input_dim}")
    print(f"  Vocab size: {vocab_size}")

    # Benchmark each configuration
    results = []

    for config in configs:
        print(f"\n{'-' * 80}")
        print(f"Testing {config['name']} Model")
        print(f"{'-' * 80}")

        # Create model
        model = RNNT(
            input_dim=input_dim,
            vocab_size=vocab_size,
            encoder_hidden=config['encoder_hidden'],
            predictor_hidden=config['predictor_hidden'],
            joint_hidden=config['joint_hidden'],
            embedding_dim=config['embedding_dim'],
            num_encoder_layers=config['num_encoder_layers'],
            num_predictor_layers=config['num_predictor_layers']
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Create dummy data
        inputs = torch.randn(batch_size, max_time, input_dim).to(device)
        input_lengths = torch.tensor([max_time] * batch_size).to(device)
        targets = torch.randint(1, vocab_size, (batch_size, max_label_length)).to(device)
        target_lengths = torch.tensor([max_label_length] * batch_size).to(device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(inputs, input_lengths, targets, target_lengths)

        # Benchmark forward pass
        num_iterations = 10
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        for _ in range(num_iterations):
            with torch.no_grad():
                logits = model(inputs, input_lengths, targets, target_lengths)
                torch.cuda.synchronize() if device == 'cuda' else None

        end_time = time.time()
        avg_forward_time = (end_time - start_time) / num_iterations

        print(f"\nPerformance:")
        print(f"  Forward pass time: {avg_forward_time * 1000:.2f} ms")
        print(f"  Throughput: {batch_size / avg_forward_time:.2f} samples/sec")

        # Benchmark with loss computation
        criterion = RNNTLoss(blank=0)

        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        for _ in range(num_iterations):
            logits = model(inputs, input_lengths, targets, target_lengths)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            torch.cuda.synchronize() if device == 'cuda' else None

        end_time = time.time()
        avg_loss_time = (end_time - start_time) / num_iterations

        print(f"  Forward + Loss time: {avg_loss_time * 1000:.2f} ms")
        print(f"  Loss computation overhead: {(avg_loss_time - avg_forward_time) * 1000:.2f} ms")

        # Benchmark backward pass
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        for _ in range(num_iterations):
            model.zero_grad()
            logits = model(inputs, input_lengths, targets, target_lengths)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            torch.cuda.synchronize() if device == 'cuda' else None

        end_time = time.time()
        avg_backward_time = (end_time - start_time) / num_iterations

        print(f"  Forward + Backward time: {avg_backward_time * 1000:.2f} ms")
        print(f"  Training throughput: {batch_size / avg_backward_time:.2f} samples/sec")

        # Memory usage
        if device == 'cuda':
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"\nMemory Usage:")
            print(f"  Peak memory: {memory_allocated:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

        # Benchmark greedy decoding
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        decoded = model.greedy_decode(inputs[:1], input_lengths[:1], max_symbols=50)

        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        decode_time = (end_time - start_time) * 1000

        print(f"\nDecoding (single sample):")
        print(f"  Greedy decode time: {decode_time:.2f} ms")
        print(f"  Decoded length: {len(decoded[0])}")

        # Store results
        results.append({
            'name': config['name'],
            'params': total_params,
            'forward_time': avg_forward_time * 1000,
            'loss_time': avg_loss_time * 1000,
            'backward_time': avg_backward_time * 1000,
            'decode_time': decode_time
        })

    # Summary table
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")

    print(f"{'Model':<10} {'Params':<15} {'Forward (ms)':<15} {'Loss (ms)':<15} {'Backward (ms)':<15} {'Decode (ms)':<15}")
    print(f"{'-' * 95}")

    for r in results:
        print(f"{r['name']:<10} {r['params']:>14,} {r['forward_time']:>14.2f} {r['loss_time']:>14.2f} {r['backward_time']:>14.2f} {r['decode_time']:>14.2f}")


def example_training_step():
    """
    Example of a single training step
    """
    print("\n" + "=" * 80)
    print("Example Training Step")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    model = RNNT(
        input_dim=80,
        vocab_size=30,
        encoder_hidden=256,
        predictor_hidden=256,
        joint_hidden=256,
        embedding_dim=128
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create loss function
    criterion = RNNTLoss(blank=0)

    # Create dummy batch
    batch_size = 4
    inputs = torch.randn(batch_size, 100, 80).to(device)
    input_lengths = torch.tensor([100, 95, 90, 85]).to(device)
    targets = torch.randint(1, 30, (batch_size, 20)).to(device)
    target_lengths = torch.tensor([20, 18, 15, 12]).to(device)

    print(f"\nBatch info:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shapes: {inputs.shape}")
    print(f"  Target shapes: {targets.shape}")

    # Training step
    model.train()
    optimizer.zero_grad()

    # Forward pass
    print("\nForward pass...")
    logits = model(inputs, input_lengths, targets, target_lengths)
    print(f"  Logits shape: {logits.shape}")

    # Compute loss
    print("\nComputing loss...")
    loss = criterion(logits, targets, input_lengths, target_lengths)
    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    print("\nBackward pass...")
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    print(f"  Gradient norm: {grad_norm:.4f}")

    # Optimizer step
    optimizer.step()
    print("  Optimizer step completed")

    print("\nTraining step completed successfully!")


def example_inference():
    """
    Example of inference with greedy decoding
    """
    print("\n" + "=" * 80)
    print("Example Inference (Greedy Decoding)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    model = RNNT(
        input_dim=80,
        vocab_size=30,
        encoder_hidden=256,
        predictor_hidden=256,
        joint_hidden=256,
        embedding_dim=128
    ).to(device)

    model.eval()

    # Create dummy input
    inputs = torch.randn(2, 100, 80).to(device)
    input_lengths = torch.tensor([100, 85]).to(device)

    print(f"\nInput info:")
    print(f"  Number of samples: 2")
    print(f"  Input lengths: {input_lengths.tolist()}")

    # Greedy decoding
    print("\nPerforming greedy decoding...")
    with torch.no_grad():
        decoded = model.greedy_decode(inputs, input_lengths, max_symbols=50)

    print(f"\nDecoded sequences:")
    for i, seq in enumerate(decoded):
        print(f"  Sample {i + 1}: {seq}")
        print(f"    Length: {len(seq)}")

    print("\nInference completed successfully!")


def main():
    """Run all examples and benchmarks"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run examples
    example_training_step()
    example_inference()

    # Run benchmarks
    benchmark_model()


if __name__ == "__main__":
    main()
