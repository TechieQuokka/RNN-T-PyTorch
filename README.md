# RNN-T (Recurrent Neural Network Transducer) Implementation

PyTorch implementation of RNN-T based on the paper "Sequence Transduction with Recurrent Neural Networks" by Alex Graves (2012).

## 📚 Paper Reference

- **Title**: Sequence Transduction with Recurrent Neural Networks
- **Author**: Alex Graves
- **Conference**: ICML 2012 Workshop on Representation Learning
- **arXiv**: [1211.3711](https://arxiv.org/abs/1211.3711)
- **PDF**: https://arxiv.org/pdf/1211.3711

## 🏗️ Architecture

RNN-T consists of three main components:

### 1. **Encoder (Transcription Network)**
- Encodes input acoustic features (e.g., mel-spectrogram) into hidden representations
- Uses LSTM layers with layer normalization
- Input: Audio features of shape `(batch, time, input_dim)`
- Output: Encoded features of shape `(batch, time, encoder_hidden)`

### 2. **Prediction Network**
- Language model that predicts next token based on previous outputs
- Uses embedding layer + LSTM layers
- Input: Previous output tokens of shape `(batch, label_length)`
- Output: Prediction features of shape `(batch, label_length, predictor_hidden)`

### 3. **Joint Network**
- Combines encoder and prediction network outputs
- Uses feedforward network with tanh activation
- Input: Encoder and predictor outputs
- Output: Logits of shape `(batch, time, label_length+1, vocab_size)`

## 📁 Project Structure

```
benchmark/
├── model.py          # RNN-T model implementation
├── loss.py           # RNN-T loss function
├── train.py          # Training script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Model Usage

```python
from model import RNNT

# Create model
model = RNNT(
    input_dim=80,          # Mel-spectrogram features
    vocab_size=30,         # Vocabulary size (including blank)
    encoder_hidden=256,
    predictor_hidden=256,
    joint_hidden=256,
    embedding_dim=128
)

# Forward pass (training)
logits = model(inputs, input_lengths, targets, target_lengths)

# Greedy decoding (inference)
decoded = model.greedy_decode(inputs, input_lengths)
```

### Training

```python
from train import Trainer, DummyASRDataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset (replace with your actual dataset)
train_dataset = DummyASRDataset(num_samples=1000)
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    learning_rate=1e-3
)

# Train
trainer.train(num_epochs=50)
```

Or run the training script directly:

```bash
python train.py
```

### Loss Computation

```python
from loss import RNNTLoss

criterion = RNNTLoss(blank=0, reduction='mean')
loss = criterion(logits, targets, input_lengths, target_lengths)
```

## 🔬 Model Components

### Encoder
```python
from model import Encoder

encoder = Encoder(
    input_dim=80,
    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)
```

### Prediction Network
```python
from model import PredictionNetwork

predictor = PredictionNetwork(
    vocab_size=30,
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)
```

### Joint Network
```python
from model import JointNetwork

joint = JointNetwork(
    encoder_dim=512,
    predictor_dim=512,
    hidden_dim=512,
    vocab_size=30
)
```

## 📊 Training Details

### Hyperparameters (Default)
- **Input Dimension**: 80 (mel-spectrogram features)
- **Vocabulary Size**: 30 (example, including blank token at index 0)
- **Encoder Hidden**: 256
- **Predictor Hidden**: 256
- **Joint Hidden**: 256
- **Embedding Dim**: 128
- **Batch Size**: 16
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW with weight decay 1e-5
- **Gradient Clipping**: Max norm 5.0
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)

### Loss Function
- Uses torchaudio's optimized `rnnt_loss` implementation
- Computes negative log-likelihood using forward-backward algorithm
- Considers all possible alignments between input and target sequences

## 🎯 Key Features

1. **Efficient Implementation**
   - Uses packed sequences for variable-length inputs
   - Layer normalization for training stability
   - Gradient clipping to prevent exploding gradients

2. **Flexible Architecture**
   - Configurable number of layers
   - Adjustable hidden dimensions
   - Dropout for regularization

3. **Inference Support**
   - Greedy decoding for fast inference
   - Supports batch processing
   - Easy to extend to beam search

4. **Training Pipeline**
   - Automatic checkpointing (best & last)
   - Learning rate scheduling
   - Validation monitoring
   - Progress bars with tqdm

## 📈 Performance Notes

- **Memory Usage**: Scales with `O(B × T × U × V)` where:
  - B = batch size
  - T = max time steps
  - U = max label length
  - V = vocabulary size

- **Training Time**: Depends on sequence lengths and model size
  - Typical training on GPU: ~1-2 seconds per batch (16 samples)

## 🔄 Customization

### Using Real Data

Replace `DummyASRDataset` with your actual speech dataset:

```python
class MyASRDataset(Dataset):
    def __init__(self, data_path):
        # Load your audio files and transcriptions
        pass

    def __getitem__(self, idx):
        # Return (audio_features, transcription)
        # audio_features: (time, feature_dim) tensor
        # transcription: (label_length,) tensor of token indices
        pass
```

### Customizing the Model

```python
# Larger model for better accuracy
model = RNNT(
    input_dim=80,
    vocab_size=5000,           # Real vocabulary size
    encoder_hidden=512,        # Increased capacity
    predictor_hidden=512,
    joint_hidden=512,
    embedding_dim=256,
    num_encoder_layers=4,      # Deeper network
    num_predictor_layers=2,
    dropout=0.1
)
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce hidden dimensions
- Use gradient accumulation
- Enable mixed precision training (AMP)

### Loss not decreasing
- Check learning rate (try 1e-4 or 1e-3)
- Verify data preprocessing
- Check for NaN values
- Ensure proper padding and masking

### Slow training
- Use GPU if available
- Increase batch size
- Reduce model size
- Use DataLoader with multiple workers

## 📖 Additional Resources

- [Original Paper (arXiv)](https://arxiv.org/abs/1211.3711)
- [Sequence-to-sequence learning with Transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- [PyTorch Audio RNN-T Tutorial](https://pytorch.org/audio/main/tutorials/online_asr_tutorial.html)
- [Assembly AI RNN-T Overview](https://www.assemblyai.com/blog/an-overview-of-transducer-models-for-asr)

## 📝 Citation

```bibtex
@article{graves2012sequence,
  title={Sequence transduction with recurrent neural networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1211.3711},
  year={2012}
}
```

## ⚠️ Notes

- The blank token is fixed at index 0
- Targets should not include the blank token
- Input/target lengths must be provided for proper masking
- For production use, consider implementing beam search decoding
- Current implementation uses greedy decoding for simplicity

## 🔮 Future Improvements

- [ ] Beam search decoding
- [ ] Attention mechanism integration
- [ ] Conformer encoder option
- [ ] Mixed precision training
- [ ] ONNX export support
- [ ] TorchScript compilation
- [ ] Streaming inference support
- [ ] Multi-GPU training
