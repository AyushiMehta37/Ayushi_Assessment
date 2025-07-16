import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import requests
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class LSTMTextGenerator:
    def __init__(self, sequence_length=40, embedding_dim=100, lstm_units=256):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.vocab_size = 0
        self.char_to_idx = {}
        self.idx_to_char = {}

    def download_dataset(self, url="https://www.gutenberg.org/files/100/100-0.txt", path=Path('shakespeare.txt')):
        try:
            r = requests.get(url)
            r.raise_for_status()
            path.write_text(r.text, encoding='utf-8')
            logging.info("Dataset downloaded.")
            return str(path)
        except Exception as e:
            logging.error(f"Download error: {e}")
            return None

    def load_and_preprocess_text(self, filepath):
        text = Path(filepath).read_text(encoding='utf-8')
        start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if start != -1 and end != -1:
            start = text.find('\n', start) + 1
            text = text[start:end]
        text = re.sub(r'\s+', ' ', text.lower())
        text = re.sub(r"[^a-z\s.,!?;:'-]", '', text)
        return text.strip()[:300_000]

    def create_mappings(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

    def create_sequences(self, text):
        idxs = [self.char_to_idx[c] for c in text]
        X = [idxs[i:i+self.sequence_length] for i in range(len(idxs) - self.sequence_length)]
        y = [idxs[i+self.sequence_length] for i in range(len(idxs) - self.sequence_length)]
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        return X, y

    def build_model(self):
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.sequence_length),
            LSTM(self.lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(self.lstm_units, dropout=0.3, recurrent_dropout=0.3),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y, epochs=20, batch_size=128, validation_split=0.2):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        return self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, callbacks=callbacks, verbose=1
        )

    def generate(self, seed, length=200, temperature=0.7):
        seed = seed.lower()
        indices = [self.char_to_idx.get(c, 0) for c in seed[-self.sequence_length:]]
        if len(indices) < self.sequence_length:
            indices = [0] * (self.sequence_length - len(indices)) + indices
        generated = seed
        for _ in range(length):
            seq = np.array(indices[-self.sequence_length:]).reshape(1, -1)
            preds = self.model.predict(seq, verbose=0)[0]
            preds = np.log(preds + 1e-8) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_idx = np.random.choice(len(preds), p=preds)
            next_char = self.idx_to_char[next_idx]
            generated += next_char
            indices.append(next_idx)
        return generated

    def save(self, model_path='lstm_generator.h5', mapping_path='tokenizer_mappings.pkl'):
        self.model.save(model_path)
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length
            }, f)

    def load(self, model_path='lstm_generator.h5', mapping_path='tokenizer_mappings.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        with open(mapping_path, 'rb') as f:
            d = pickle.load(f)
            self.char_to_idx = d['char_to_idx']
            self.idx_to_char = d['idx_to_char']
            self.vocab_size = d['vocab_size']
            self.sequence_length = d['sequence_length']

def main():
    logging.info("LSTM Text Generation System")
    gen = LSTMTextGenerator(sequence_length=40, embedding_dim=100, lstm_units=256)
    dataset_file = Path('shakespeare.txt')
    if not dataset_file.exists():
        logging.info(f"Dataset file '{dataset_file}' not found. Downloading...")
        if not gen.download_dataset():
            logging.error("Failed to download dataset.")
            return
    text = gen.load_and_preprocess_text(dataset_file)
    logging.info(f"Text length: {len(text)} characters")
    gen.create_mappings(text)
    logging.info(f"Vocabulary size: {gen.vocab_size}")
    X, y = gen.create_sequences(text)
    logging.info(f"Training sequences: {X.shape[0]}")
    gen.build_model()
    gen.model.summary()
    gen.train(X, y, epochs=20, batch_size=128)
    gen.save()
    logging.info("\nGenerated Text Samples:\n")
    seeds = ["to be or not to be", "hamlet", "romeo and juliet", "once upon a time", "the king"]
    for seed in seeds:
        logging.info(f"Seed: '{seed}'")
        generated = gen.generate(seed, length=150, temperature=0.7)
        logging.info(f"Generated: {generated}\n{'-'*50}")

if __name__ == "__main__":
    main()