#!/usr/bin/env python3
"""
Neural Network Pretraining Module for MCP Core System
Handles pretraining of various neural network models used in the system.
"""

import logging
import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Try to import optional dependencies
try:
    from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
    from transformers import DataCollatorWithPadding
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class PretrainConfig:
    """Configuration for neural network pretraining."""
    
    def __init__(self, 
                 model_type: str = "embedding",
                 base_model: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 learning_rate: float = 2e-5,
                 epochs: int = 3,
                 max_seq_length: int = 128,
                 train_test_split: float = 0.8,
                 device: str = None,
                 output_dir: str = "data/models",
                 cache_dir: str = "data/cache",
                 save_steps: int = 500,
                 eval_steps: int = 100,
                 log_level: str = "INFO"):
        """Initialize pretraining configuration."""
        self.model_type = model_type
        self.base_model = base_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.train_test_split = train_test_split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.log_level = log_level

class MemoryDataset(Dataset):
    """Dataset for training with memory data."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 128):
        """Initialize memory dataset."""
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'label': label
            }

class SimilarityDataset(Dataset):
    """Dataset for training with text similarity data."""
    
    def __init__(self, text_pairs: List[Tuple[str, str]], 
                 scores: List[float], tokenizer=None, max_length: int = 128):
        """Initialize similarity dataset."""
        self.text_pairs = text_pairs
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text_pairs)
    
    def __getitem__(self, idx):
        text1, text2 = self.text_pairs[idx]
        score = self.scores[idx]
        
        if self.tokenizer:
            encoding1 = self.tokenizer(
                text1,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            encoding2 = self.tokenizer(
                text2,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                'input_ids1': encoding1['input_ids'].flatten(),
                'attention_mask1': encoding1['attention_mask'].flatten(),
                'input_ids2': encoding2['input_ids'].flatten(),
                'attention_mask2': encoding2['attention_mask'].flatten(),
                'score': torch.tensor(score, dtype=torch.float)
            }
        else:
            return {
                'text1': text1,
                'text2': text2,
                'score': score
            }

class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model for text representation."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        """Initialize simple embedding model."""
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, x):
        """Forward pass."""
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

class NeuralNetworkPretrainer:
    """
    Neural Network Pretrainer for MCP Core System.
    
    Handles pretraining of various neural network models:
    - Embedding models for semantic search
    - Classification models for categorization
    - Similarity models for comparison
    """
    
    def __init__(self, config: PretrainConfig = None):
        """Initialize neural network pretrainer."""
        self.config = config or PretrainConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create output and cache directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.training_progress = {
            'status': 'idle',
            'model_type': None,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_step': 0,
            'total_steps': 0,
            'loss': 0.0,
            'eval_loss': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_memory_data(self, memory_manager) -> Tuple[List[str], List[str]]:
        """Load training data from memory manager."""
        try:
            # Get memories from memory manager
            memories = memory_manager.search_memories("", limit=10000)
            
            texts = []
            types = []
            
            for memory in memories:
                text = memory.get('text', '')
                memory_type = memory.get('memory_type', 'general')
                
                if text and len(text.strip()) > 10:  # Filter out very short texts
                    texts.append(text)
                    types.append(memory_type)
            
            self.logger.info(f"Loaded {len(texts)} memory items for training")
            return texts, types
            
        except Exception as e:
            self.logger.error(f"Error loading memory data: {e}")
            return [], []
    
    def _generate_similarity_data(self, texts: List[str]) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Generate text similarity data for training."""
        import random
        from difflib import SequenceMatcher
        
        text_pairs = []
        scores = []
        
        # Generate similar pairs (with high scores)
        for i in range(min(1000, len(texts))):
            text = texts[i]
            # Create a slightly modified version
            words = text.split()
            if len(words) > 5:
                # Remove or replace some words
                for _ in range(min(3, len(words) // 5)):
                    idx = random.randint(0, len(words) - 1)
                    if random.random() < 0.5:
                        words.pop(idx)
                    else:
                        words[idx] = random.choice(["the", "a", "an", "this", "that", "some", "very"])
                
                modified_text = " ".join(words)
                similarity = SequenceMatcher(None, text, modified_text).ratio()
                
                text_pairs.append((text, modified_text))
                scores.append(similarity)
        
        # Generate dissimilar pairs (with low scores)
        for i in range(min(1000, len(texts))):
            text1 = texts[i]
            text2 = texts[random.randint(0, len(texts) - 1)]
            
            similarity = SequenceMatcher(None, text1, text2).ratio()
            if similarity < 0.5:  # Only include if they're sufficiently different
                text_pairs.append((text1, text2))
                scores.append(similarity)
        
        self.logger.info(f"Generated {len(text_pairs)} text pairs for similarity training")
        return text_pairs, scores
    
    def pretrain_embedding_model(self, memory_manager=None, texts: List[str] = None):
        """
        Pretrain an embedding model for semantic search.
        
        Args:
            memory_manager: Optional memory manager to load data from
            texts: Optional list of texts to train on
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("Sentence Transformers not available. Please install with: pip install sentence-transformers")
            return False
        
        try:
            # Set training state
            self.training_progress = {
                'status': 'initializing',
                'model_type': 'embedding',
                'current_epoch': 0,
                'total_epochs': self.config.epochs,
                'current_step': 0,
                'total_steps': 0,
                'loss': 0.0,
                'eval_loss': 0.0,
                'start_time': time.time(),
                'end_time': None
            }
            
            # Load training data
            if texts is None and memory_manager is not None:
                texts, _ = self._load_memory_data(memory_manager)
            
            if not texts or len(texts) < 100:
                self.logger.error("Insufficient training data")
                self.training_progress['status'] = 'failed'
                return False
            
            # Generate text pairs for contrastive learning
            text_pairs, scores = self._generate_similarity_data(texts)
            
            # Create sentence transformer model
            model_name = self.config.base_model
            model_save_path = os.path.join(self.config.output_dir, f"embedding-{int(time.time())}")
            
            model = SentenceTransformer(model_name, cache_folder=self.config.cache_dir)
            
            # Prepare training data
            train_examples = [
                InputExample(texts=[text1, text2], label=score)
                for (text1, text2), score in zip(text_pairs, scores)
            ]
            
            # Split into train and validation
            train_size = int(len(train_examples) * self.config.train_test_split)
            train_data = train_examples[:train_size]
            val_data = train_examples[train_size:]
            
            # Define training arguments
            train_dataloader = DataLoader(
                train_data, 
                shuffle=True, 
                batch_size=self.config.batch_size
            )
            
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Create evaluator
            evaluator = None
            if val_data:
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    val_data, name='embedding-eval'
                )
            
            # Update training state
            self.training_progress['status'] = 'training'
            self.training_progress['total_steps'] = len(train_dataloader) * self.config.epochs
            
            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.config.epochs,
                evaluation_steps=self.config.eval_steps,
                warmup_steps=100,
                output_path=model_save_path,
                show_progress_bar=True,
                callback=self._training_callback
            )
            
            # Save model
            model.save(model_save_path)
            
            # Store model in cache
            self.models['embedding'] = model
            
            # Update training state
            self.training_progress['status'] = 'completed'
            self.training_progress['end_time'] = time.time()
            
            self.logger.info(f"Embedding model training completed and saved to {model_save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error pretraining embedding model: {e}")
            self.training_progress['status'] = 'failed'
            return False
    
    def pretrain_classification_model(self, memory_manager=None, 
                                     texts: List[str] = None, 
                                     labels: List[int] = None):
        """
        Pretrain a classification model.
        
        Args:
            memory_manager: Optional memory manager to load data from
            texts: Optional list of texts to train on
            labels: Optional list of labels for classification
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers not available. Please install with: pip install transformers")
            return False
        
        try:
            # Set training state
            self.training_progress = {
                'status': 'initializing',
                'model_type': 'classification',
                'current_epoch': 0,
                'total_epochs': self.config.epochs,
                'current_step': 0,
                'total_steps': 0,
                'loss': 0.0,
                'eval_loss': 0.0,
                'start_time': time.time(),
                'end_time': None
            }
            
            # Load training data
            if texts is None and memory_manager is not None:
                texts, types = self._load_memory_data(memory_manager)
                
                # Convert types to numeric labels
                unique_types = list(set(types))
                type_to_label = {t: i for i, t in enumerate(unique_types)}
                labels = [type_to_label[t] for t in types]
            
            if not texts or len(texts) < 100 or not labels:
                self.logger.error("Insufficient training data")
                self.training_progress['status'] = 'failed'
                return False
            
            # Create tokenizer and model
            model_name = self.config.base_model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.config.cache_dir
            )
            
            from transformers import AutoModelForSequenceClassification
            
            num_labels = len(set(labels))
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=self.config.cache_dir
            )
            
            # Create dataset
            dataset = MemoryDataset(texts, labels, tokenizer, self.config.max_seq_length)
            
            # Split into train and validation
            train_size = int(len(dataset) * self.config.train_test_split)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Define training arguments
            model_save_path = os.path.join(self.config.output_dir, f"classification-{int(time.time())}")
            
            training_args = TrainingArguments(
                output_dir=model_save_path,
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=os.path.join(self.config.output_dir, "logs"),
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                load_best_model_at_end=True
            )
            
            # Create data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # Update training state
            self.training_progress['status'] = 'training'
            self.training_progress['total_steps'] = len(train_dataset) // self.config.batch_size * self.config.epochs
            
            # Train the model
            trainer.train()
            
            # Save model and tokenizer
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Save label mapping
            if memory_manager is not None:
                label_mapping = {i: t for t, i in type_to_label.items()}
                with open(os.path.join(model_save_path, "label_mapping.json"), "w") as f:
                    json.dump(label_mapping, f)
            
            # Store model in cache
            self.models['classification'] = model
            self.tokenizers['classification'] = tokenizer
            
            # Update training state
            self.training_progress['status'] = 'completed'
            self.training_progress['end_time'] = time.time()
            
            self.logger.info(f"Classification model training completed and saved to {model_save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error pretraining classification model: {e}")
            self.training_progress['status'] = 'failed'
            return False
    
    def pretrain_similarity_model(self, memory_manager=None, texts: List[str] = None):
        """
        Pretrain a similarity model for text comparison.
        
        Args:
            memory_manager: Optional memory manager to load data from
            texts: Optional list of texts to train on
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("Sentence Transformers not available. Please install with: pip install sentence-transformers")
            return False
        
        try:
            # Set training state
            self.training_progress = {
                'status': 'initializing',
                'model_type': 'similarity',
                'current_epoch': 0,
                'total_epochs': self.config.epochs,
                'current_step': 0,
                'total_steps': 0,
                'loss': 0.0,
                'eval_loss': 0.0,
                'start_time': time.time(),
                'end_time': None
            }
            
            # Load training data
            if texts is None and memory_manager is not None:
                texts, _ = self._load_memory_data(memory_manager)
            
            if not texts or len(texts) < 100:
                self.logger.error("Insufficient training data")
                self.training_progress['status'] = 'failed'
                return False
            
            # Generate text pairs for similarity learning
            text_pairs, scores = self._generate_similarity_data(texts)
            
            # Create sentence transformer model
            model_name = self.config.base_model
            model_save_path = os.path.join(self.config.output_dir, f"similarity-{int(time.time())}")
            
            model = SentenceTransformer(model_name, cache_folder=self.config.cache_dir)
            
            # Prepare training data
            train_examples = [
                InputExample(texts=[text1, text2], label=score)
                for (text1, text2), score in zip(text_pairs, scores)
            ]
            
            # Split into train and validation
            train_size = int(len(train_examples) * self.config.train_test_split)
            train_data = train_examples[:train_size]
            val_data = train_examples[train_size:]
            
            # Define training arguments
            train_dataloader = DataLoader(
                train_data, 
                shuffle=True, 
                batch_size=self.config.batch_size
            )
            
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Create evaluator
            evaluator = None
            if val_data:
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    val_data, name='similarity-eval'
                )
            
            # Update training state
            self.training_progress['status'] = 'training'
            self.training_progress['total_steps'] = len(train_dataloader) * self.config.epochs
            
            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.config.epochs,
                evaluation_steps=self.config.eval_steps,
                warmup_steps=100,
                output_path=model_save_path,
                show_progress_bar=True,
                callback=self._training_callback
            )
            
            # Save model
            model.save(model_save_path)
            
            # Store model in cache
            self.models['similarity'] = model
            
            # Update training state
            self.training_progress['status'] = 'completed'
            self.training_progress['end_time'] = time.time()
            
            self.logger.info(f"Similarity model training completed and saved to {model_save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error pretraining similarity model: {e}")
            self.training_progress['status'] = 'failed'
            return False
    
    def pretrain_all_models(self, memory_manager=None, async_mode: bool = True):
        """
        Pretrain all neural network models.
        
        Args:
            memory_manager: Optional memory manager to load data from
            async_mode: Whether to run training asynchronously
        """
        if self.is_training:
            self.logger.warning("Training already in progress")
            return False
        
        def train_all():
            try:
                self.is_training = True
                
                # Load training data once
                texts = None
                types = None
                if memory_manager is not None:
                    texts, types = self._load_memory_data(memory_manager)
                
                # Train embedding model
                self.pretrain_embedding_model(texts=texts)
                
                # Train classification model
                if texts and types:
                    unique_types = list(set(types))
                    type_to_label = {t: i for i, t in enumerate(unique_types)}
                    labels = [type_to_label[t] for t in types]
                    self.pretrain_classification_model(texts=texts, labels=labels)
                
                # Train similarity model
                self.pretrain_similarity_model(texts=texts)
                
            finally:
                self.is_training = False
        
        if async_mode:
            # Run training in a separate thread
            self.training_thread = threading.Thread(target=train_all)
            self.training_thread.daemon = True
            self.training_thread.start()
            return True
        else:
            # Run training synchronously
            train_all()
            return True
    
    def load_model(self, model_type: str, model_path: str):
        """
        Load a pretrained model.
        
        Args:
            model_type: Type of model ('embedding', 'classification', 'similarity')
            model_path: Path to the model directory
        """
        try:
            if model_type == 'embedding' or model_type == 'similarity':
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.logger.error("Sentence Transformers not available")
                    return False
                
                model = SentenceTransformer(model_path)
                self.models[model_type] = model
                self.logger.info(f"Loaded {model_type} model from {model_path}")
                return True
                
            elif model_type == 'classification':
                if not TRANSFORMERS_AVAILABLE:
                    self.logger.error("Transformers not available")
                    return False
                
                from transformers import AutoModelForSequenceClassification
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                self.models[model_type] = model
                self.tokenizers[model_type] = tokenizer
                
                # Load label mapping if available
                label_mapping_path = os.path.join(model_path, "label_mapping.json")
                if os.path.exists(label_mapping_path):
                    with open(label_mapping_path, "r") as f:
                        self.label_mapping = json.load(f)
                
                self.logger.info(f"Loaded classification model from {model_path}")
                return True
            
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available pretrained models."""
        models = {
            'embedding': [],
            'classification': [],
            'similarity': []
        }
        
        try:
            for model_type in models.keys():
                model_prefix = f"{model_type}-"
                for item in os.listdir(self.config.output_dir):
                    item_path = os.path.join(self.config.output_dir, item)
                    if os.path.isdir(item_path) and item.startswith(model_prefix):
                        models[model_type].append(item)
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
        
        return models
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        progress = self.training_progress.copy()
        
        # Add elapsed time if training is in progress
        if progress['status'] in ['initializing', 'training'] and progress['start_time']:
            progress['elapsed_time'] = time.time() - progress['start_time']
        elif progress['end_time'] and progress['start_time']:
            progress['elapsed_time'] = progress['end_time'] - progress['start_time']
        
        return progress
    
    def _training_callback(self, score, epoch, steps):
        """Callback for training progress updates."""
        self.training_progress['current_epoch'] = epoch
        self.training_progress['current_step'] = steps
        self.training_progress['eval_loss'] = score
        
        self.logger.debug(f"Training progress: Epoch {epoch}, Step {steps}, Score {score}")

# Convenience functions

def create_pretrainer(config: Optional[PretrainConfig] = None) -> NeuralNetworkPretrainer:
    """Create a neural network pretrainer with custom configuration."""
    return NeuralNetworkPretrainer(config or PretrainConfig())

def get_default_pretrainer() -> NeuralNetworkPretrainer:
    """Get a default neural network pretrainer."""
    return NeuralNetworkPretrainer()

async def pretrain_models(memory_manager=None, model_types: List[str] = None):
    """
    Pretrain specified neural network models.
    
    Args:
        memory_manager: Optional memory manager to load data from
        model_types: List of model types to train ('embedding', 'classification', 'similarity')
                    If None, all models will be trained
    """
    pretrainer = get_default_pretrainer()
    
    if not model_types:
        model_types = ['embedding', 'classification', 'similarity']
    
    results = {}
    
    for model_type in model_types:
        if model_type == 'embedding':
            results[model_type] = pretrainer.pretrain_embedding_model(memory_manager)
        elif model_type == 'classification':
            results[model_type] = pretrainer.pretrain_classification_model(memory_manager)
        elif model_type == 'similarity':
            results[model_type] = pretrainer.pretrain_similarity_model(memory_manager)
    
    return results