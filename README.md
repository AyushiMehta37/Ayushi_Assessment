# LSTM Text Generator

Character-level text generation using LSTM neural networks with TensorFlow/Keras. Trains on Shakespeare's complete works from Project Gutenberg.

## Requirements

```bash
pip install numpy tensorflow requests
```

## Architecture

- **Embedding layer**: Character indices → dense vectors (100D)
- **Two LSTM layers**: 256 units each with 0.3 dropout
- **Dense output**: Softmax for character probabilities

## Usage

### Basic Training
```python
python lstm_text_generator.py
```

### Custom Training
```python
from lstm_text_generator import LSTMTextGenerator

gen = LSTMTextGenerator(sequence_length=50, embedding_dim=128, lstm_units=512)
text = gen.load_and_preprocess_text('your_text.txt')
gen.create_mappings(text)
X, y = gen.create_sequences(text)
gen.build_model()
gen.train(X, y, epochs=30, batch_size=64)
gen.save('my_model.h5', 'my_mappings.pkl')
```

### Text Generation
```python
gen = LSTMTextGenerator()
gen.load('lstm_generator.h5', 'tokenizer_mappings.pkl')
text = gen.generate("to be or not to be", length=200, temperature=0.7)
```

## Parameters

**Model**: `sequence_length` (40), `embedding_dim` (100), `lstm_units` (256)  
**Training**: `epochs` (20), `batch_size` (128), `validation_split` (0.2)  
**Generation**: `temperature` - Low (0.1-0.5): conservative, High (0.9-2.0): creative

## Files Generated

- `shakespeare.txt` - Downloaded dataset
- `lstm_generator.h5` - Trained model
- `tokenizer_mappings.pkl` - Character mappings
- `best_model.h5` - Best checkpoint during training

## Training Process

1. Downloads Shakespeare's complete works from Project Gutenberg
2. Preprocesses text: lowercase, removes special characters, limits to 300k chars
3. Creates overlapping character sequences
4. Trains with early stopping and checkpointing

## Troubleshooting

- **Out of Memory**: Reduce batch size or sequence length
- **Poor Output**: Train longer or increase model capacity  
- **Repetitive Text**: Lower temperature
- **Download Fails**: Check internet connection
