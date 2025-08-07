import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Optional
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SubtleToxicityTagger:
    """Implementation of SUBTLE_TOXICITY tagging method."""

    def __init__(self):
        # Passive-aggressive patterns
        self.passive_aggressive_patterns = [
            r"\bjust saying\b", r"\bno offense\b", r"\bwith all due respect\b",
            r"\bi'm just\b", r"\bbless your heart\b", r"\bgood for you\b",
            r"\bwhatever\b", r"\bif you say so\b", r"\bsure thing\b", r"\bokay then\b"
        ]

        # Sarcastic indicators
        self.sarcastic_patterns = [
            r"\boh really\b", r"\bhow original\b", r"\bwow\b.*\bgenius\b",
            r"\bbrilliant\b", r"\bfascinating\b", r"\bimpressive\b",
            r"\bcongratulations\b", r"\bgood job\b", r"\bwell done\b", r"\bobviously\b"
        ]

        # Dismissive language
        self.dismissive_patterns = [
            r"\byou don't understand\b", r"\byou wouldn't get it\b", r"\bnever mind\b",
            r"\bforget it\b", r"\bdon't bother\b", r"\bwaste of time\b",
            r"\bpointless\b", r"\buseless\b", r"\bwhatever\b", r"\bwho cares\b"
        ]

        # Condescending phrases
        self.condescending_patterns = [
            r"\blet me explain\b", r"\byou need to understand\b", r"\bactually\b",
            r"\bwell actually\b", r"\bfor your information\b", r"\bfyi\b",
            r"\bas i said\b", r"\blike i said\b", r"\bclearly\b", r"\bobviously\b"
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = {
            'passive_aggressive': [re.compile(p, re.IGNORECASE) for p in self.passive_aggressive_patterns],
            'sarcastic': [re.compile(p, re.IGNORECASE) for p in self.sarcastic_patterns],
            'dismissive': [re.compile(p, re.IGNORECASE) for p in self.dismissive_patterns],
            'condescending': [re.compile(p, re.IGNORECASE) for p in self.condescending_patterns]
        }

    def detect_patterns(self, text: str) -> Dict[str, int]:
        """Detect subtle toxicity patterns in text."""
        pattern_counts = {
            'passive_aggressive': 0, 'sarcastic': 0,
            'dismissive': 0, 'condescending': 0
        }

        for pattern_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                pattern_counts[pattern_type] += len(matches)

        return pattern_counts

    def calculate_intensity(self, pattern_counts: Dict[str, int], text_length: int) -> str:
        """Calculate subtle toxicity intensity."""
        total_patterns = sum(pattern_counts.values())

        if total_patterns == 0:
            return "NONE"

        # Normalize by text length
        if text_length > 0:
            pattern_density = (total_patterns / text_length) * 100
        else:
            pattern_density = 0

        # Determine intensity
        if total_patterns >= 3 or pattern_density > 5:
            return "HIGH"
        elif total_patterns >= 1 or pattern_density > 2:
            return "MEDIUM"
        else:
            return "NONE"

    def apply_tagging(self, text: str) -> str:
        """Apply subtle toxicity tagging to text."""
        if not text or not isinstance(text, str):
            return str(text) if text else ""

        # Detect patterns
        pattern_counts = self.detect_patterns(text)

        # Calculate intensity
        intensity = self.calculate_intensity(pattern_counts, len(text))

        # Apply tagging if patterns are detected
        if intensity != "NONE":
            # Add intensity tag
            tagged_text = f"[SUBTLE_TOXICITY:{intensity}] {text}"

            # Add specific pattern tags
            pattern_tags = []
            for pattern_type, count in pattern_counts.items():
                if count > 0:
                    pattern_tags.append(f"[{pattern_type.upper()}]")

            if pattern_tags:
                tagged_text = " ".join(pattern_tags) + " " + tagged_text

            return tagged_text

        return text

class ToxicCommentPipeline:
    """Complete pipeline for toxic comment classification using MLflow models."""

    def __init__(self, device: str = 'auto'):
        """
        Initialize the pipeline.

        Args:
            device: 'auto', 'cpu', 'cuda', or specific device
        """
        # Configure device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Configuration from your config.json
        self.model_name = 'unitary/toxic-bert'
        self.max_length = 128  # Using 128 for efficiency, can be up to 512
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        # Initialize components
        self.tokenizer = None
        self.baseline_model = None
        self.mlflow_model = None  # Your MLflow model
        self.subtle_tagger = SubtleToxicityTagger()

        # Load models
        self._load_models()

    def _load_models(self):
        """Load tokenizer and models."""
        print("Loading baseline tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load baseline model
        self.baseline_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)

        print("Models loaded successfully!")

    def load_mlflow_model(self, model_directory: str):
        """
        Load your MLflow fine-tuned model from directory containing the files.
        
        Args:
            model_directory: Path to directory containing config.json, tokenizer files, etc.
        """
        try:
            print(f"Loading MLflow model from: {model_directory}")
            
            # Convert to Path object and resolve to handle Windows paths properly
            model_path = Path(model_directory).resolve()
            
            # Check if the directory exists and contains required files
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory does not exist: {model_path}")
            
            config_file = model_path / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"config.json not found in {model_path}")
            
            print(f"Loading model from resolved path: {model_path}")
            
            # Load the model using the local path
            self.mlflow_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),  # Convert Path back to string
                local_files_only=True,  # Force local loading
                num_labels=len(self.labels),
                problem_type="multi_label_classification"
            ).to(self.device)
            
            # Also load the tokenizer from the same directory if available
            try:
                mlflow_tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True
                )
                self.tokenizer = mlflow_tokenizer
                print("Using MLflow tokenizer")
            except Exception as tokenizer_error:
                print(f"MLflow tokenizer loading failed: {tokenizer_error}")
                print("Using baseline tokenizer")
            
            print("MLflow model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading MLflow model from {model_directory}: {e}")
            print("Continuing with baseline model only...")

    def preprocess_text(self, text: str, apply_tagging: bool = False) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for the model.

        Args:
            text: Input text
            apply_tagging: Whether to apply subtle toxicity tagging

        Returns:
            Dict with input_ids and attention_mask
        """
        if apply_tagging:
            text = self.subtle_tagger.apply_tagging(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def predict_single(self,
                      text: str,
                      model_type: str = 'baseline',
                      return_probabilities: bool = True) -> Dict[str, float]:
        """
        Predict toxicity for a single text.

        Args:
            text: Text to analyze
            model_type: 'baseline', 'mlflow', 'mlflow_tagged', or 'ensemble'
            return_probabilities: Whether to return probabilities or logits

        Returns:
            Dict with probabilities/logits for each label
        """
        # Select model
        if model_type == 'baseline':
            model = self.baseline_model
            apply_tagging = False
        elif model_type == 'mlflow':
            if self.mlflow_model is None:
                print("MLflow model not available, using baseline")
                model = self.baseline_model
            else:
                model = self.mlflow_model
            apply_tagging = False
        elif model_type == 'mlflow_tagged':
            if self.mlflow_model is None:
                print("MLflow model not available, using baseline with tagging")
                model = self.baseline_model
            else:
                model = self.mlflow_model
            apply_tagging = True
        elif model_type == 'ensemble':
            return self._predict_ensemble(text, return_probabilities)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Preprocess
        inputs = self.preprocess_text(text, apply_tagging)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]

            if return_probabilities:
                probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
                return dict(zip(self.labels, probabilities))
            else:
                return dict(zip(self.labels, logits))

    def predict_batch(self,
                     texts: List[str],
                     model_type: str = 'baseline',
                     batch_size: int = 32,
                     return_probabilities: bool = True) -> pd.DataFrame:
        """
        Predict toxicity for multiple texts.

        Args:
            texts: List of texts to analyze
            model_type: Type of model to use
            batch_size: Batch size
            return_probabilities: Whether to return probabilities or logits

        Returns:
            DataFrame with predictions
        """
        # Select model
        if model_type == 'baseline':
            model = self.baseline_model
            apply_tagging = False
        elif model_type == 'mlflow':
            model = self.mlflow_model if self.mlflow_model else self.baseline_model
            apply_tagging = False
        elif model_type == 'mlflow_tagged':
            model = self.mlflow_model if self.mlflow_model else self.baseline_model
            apply_tagging = True
        elif model_type == 'ensemble':
            return self._predict_batch_ensemble(texts, batch_size, return_probabilities)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        all_predictions = []
        model.eval()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Preprocess batch
            if apply_tagging:
                batch_texts = [self.subtle_tagger.apply_tagging(text) for text in batch_texts]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu().numpy()

                if return_probabilities:
                    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
                    all_predictions.append(probabilities)
                else:
                    all_predictions.append(logits)

        # Concatenate results
        predictions = np.vstack(all_predictions)

        # Create DataFrame
        df = pd.DataFrame(predictions, columns=self.labels)
        df['text'] = texts

        return df

    def _predict_ensemble(self, text: str, return_probabilities: bool = True) -> Dict[str, float]:
        """Ensemble prediction combining available models."""
        predictions = []
        weights = []

        # Baseline
        pred_baseline = self.predict_single(text, 'baseline', return_probabilities)
        predictions.append(list(pred_baseline.values()))
        weights.append(0.4)

        # MLflow model
        if self.mlflow_model is not None:
            pred_mlflow = self.predict_single(text, 'mlflow', return_probabilities)
            predictions.append(list(pred_mlflow.values()))
            weights.append(0.6)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return dict(zip(self.labels, ensemble_pred))

    def _predict_batch_ensemble(self, texts: List[str], batch_size: int, return_probabilities: bool) -> pd.DataFrame:
        """Ensemble prediction for batch."""
        # Get predictions from baseline
        pred_baseline = self.predict_batch(texts, 'baseline', batch_size, return_probabilities)
        ensemble_preds = pred_baseline[self.labels].values * 0.4

        # Add MLflow model if available
        if self.mlflow_model is not None:
            pred_mlflow = self.predict_batch(texts, 'mlflow', batch_size, return_probabilities)
            ensemble_preds += pred_mlflow[self.labels].values * 0.6

        # Create result DataFrame
        df = pd.DataFrame(ensemble_preds, columns=self.labels)
        df['text'] = texts

        return df

    def analyze_comment(self, text: str, detailed: bool = True) -> Dict:
        """
        Complete analysis of a comment.

        Args:
            text: Text to analyze
            detailed: Whether to include detailed analysis

        Returns:
            Dict with complete analysis
        """
        result = {
            'original_text': text,
            'predictions': {}
        }

        # Predictions from available models
        result['predictions']['baseline'] = self.predict_single(text, 'baseline')

        if self.mlflow_model is not None:
            result['predictions']['mlflow'] = self.predict_single(text, 'mlflow')
            result['predictions']['mlflow_tagged'] = self.predict_single(text, 'mlflow_tagged')

        # Ensemble if MLflow model available
        if self.mlflow_model is not None:
            result['predictions']['ensemble'] = self._predict_ensemble(text)

        if detailed:
            # Subtle pattern analysis
            pattern_counts = self.subtle_tagger.detect_patterns(text)
            intensity = self.subtle_tagger.calculate_intensity(pattern_counts, len(text))

            result['subtle_toxicity_analysis'] = {
                'intensity': intensity,
                'patterns_detected': pattern_counts,
                'tagged_text': self.subtle_tagger.apply_tagging(text)
            }

            # Text statistics
            result['text_stats'] = {
                'length': len(text),
                'word_count': len(text.split()),
                'has_caps': text.isupper(),
                'has_special_chars': bool(re.search(r'[!@#$%^&*()_+{}|:"<>?]', text))
            }

        return result

    def get_top_predictions(self, text: str, model_type: str = 'ensemble', top_k: int = 3) -> List[tuple]:
        """
        Get top-k highest predictions.

        Args:
            text: Text to analyze
            model_type: Type of model
            top_k: Number of top predictions

        Returns:
            List of tuples (label, probability)
        """
        predictions = self.predict_single(text, model_type)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]

# Utility function to create the pipeline
def create_toxic_pipeline(device: str = 'auto',
                         mlflow_model_path: Optional[str] = None) -> ToxicCommentPipeline:
    """
    Create toxicity classification pipeline with MLflow model.

    Args:
        device: Device to use
        mlflow_model_path: Path to directory containing MLflow model files

    Returns:
        Configured pipeline
    """
    pipeline = ToxicCommentPipeline(device=device)

    if mlflow_model_path:
        pipeline.load_mlflow_model(mlflow_model_path)

    return pipeline

# Example usage
if __name__ == "__main__":
    # Create pipeline with your MLflow model
    # Use the correct path to your model directory
    pipeline = create_toxic_pipeline(
        mlflow_model_path=r"C:\wd\wd_demos\toxic_comment_classification\mlruns\models\toxic_bert_finetuned_tagged"
    )

    # Example texts
    test_texts = [
        "This is a normal comment.",
        "You are such an idiot!",
        "Well actually, you don't understand how this works.",
        "I'm going to kill you!",
        "Whatever, good for you I guess."
    ]

    print("=== COMMENT ANALYSIS WITH MLFLOW MODEL ===\n")

    for i, text in enumerate(test_texts, 1):
        print(f"{i}. Text: '{text}'")

        # Complete analysis
        analysis = pipeline.analyze_comment(text, detailed=True)

        # Show baseline vs MLflow predictions
        if 'baseline' in analysis['predictions']:
            baseline_preds = analysis['predictions']['baseline']
            print("   Baseline predictions:")
            for label, prob in baseline_preds.items():
                if prob > 0.1:
                    print(f"     {label}: {prob:.3f}")

        if 'mlflow' in analysis['predictions']:
            mlflow_preds = analysis['predictions']['mlflow']
            print("   MLflow predictions:")
            for label, prob in mlflow_preds.items():
                if prob > 0.1:
                    print(f"     {label}: {prob:.3f}")

        # Show ensemble if available
        if 'ensemble' in analysis['predictions']:
            ensemble_preds = analysis['predictions']['ensemble']
            print("   Ensemble predictions:")
            for label, prob in ensemble_preds.items():
                if prob > 0.1:
                    print(f"     {label}: {prob:.3f}")

        print()