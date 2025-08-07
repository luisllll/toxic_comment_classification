
# Fine-tuning Extension for Existing Toxic Comment Classification Setup
# ====================================================================
# 
# This module integrates with the existing code structure and provides
# fine-tuning capabilities for baseline vs SUBTLE_TOXICITY comparison.
# 
# Author: Data Science Team  
# Version: 2.1 (Integrated)
# Date: August 2025

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ToxicCommentDataset(Dataset):
    """Custom dataset for toxic comment classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

class FineTuningManager:
    """Manager class for fine-tuning experiments with existing setup."""

    def __init__(self, tagger_instance, labels, device):
        self.tagger = tagger_instance
        self.labels = labels
        self.device = device
        self.tokenizer = tagger_instance.tokenizer
        self.base_model_name = tagger_instance.base_model_name

        # Training history storage
        self.training_history = {}

        print(f"FineTuningManager initialized with {self.base_model_name} on {self.device}")

    def prepare_training_data(self, X_train, y_train, X_val, y_val, method='baseline'):
        """
        Prepare training data with optional tagging method applied.

        Args:
            X_train: Training texts (numpy array or list)
            y_train: Training labels (numpy array)
            X_val: Validation texts (numpy array or list)
            y_val: Validation labels (numpy array)
            method: 'baseline' or 'subtle_toxicity'

        Returns:
            Dictionary with processed train/val data
        """
        print(f"\n{'='*60}")
        print(f"PREPARING TRAINING DATA: {method.upper()}")
        print(f"{'='*60}")

        # Convert to lists if needed
        if isinstance(X_train, np.ndarray):
            X_train = X_train.tolist()
        if isinstance(X_val, np.ndarray):
            X_val = X_val.tolist()

        # Apply tagging method
        if method == 'subtle_toxicity':
            print("Applying SUBTLE_TOXICITY tagging to training data...")
            processed_train_texts = self.tagger.apply_tagging_method(X_train, 'subtle_toxicity')
            processed_val_texts = self.tagger.apply_tagging_method(X_val, 'subtle_toxicity')

            # Show tagging statistics
            train_tagged = sum(1 for orig, tagged in zip(X_train, processed_train_texts) if orig != tagged)
            val_tagged = sum(1 for orig, tagged in zip(X_val, processed_val_texts) if orig != tagged)

            print(f"Training: Tagged {train_tagged}/{len(X_train)} texts ({train_tagged/len(X_train)*100:.1f}%)")
            print(f"Validation: Tagged {val_tagged}/{len(X_val)} texts ({val_tagged/len(X_val)*100:.1f}%)")

        elif method == 'baseline':
            processed_train_texts = X_train
            processed_val_texts = X_val
            print(f"Using baseline (no tagging)")
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Data prepared:")
        print(f"  Train: {len(processed_train_texts)} samples")
        print(f"  Validation: {len(processed_val_texts)} samples")

        return {
            'train': {'texts': processed_train_texts, 'labels': y_train},
            'val': {'texts': processed_val_texts, 'labels': y_val},
            'method': method
        }

    def create_data_loaders(self, data_dict, batch_size=16, max_length=256):
        """Create PyTorch data loaders from prepared data."""

        datasets = {}
        loaders = {}

        for split in ['train', 'val']:
            datasets[split] = ToxicCommentDataset(
                texts=data_dict[split]['texts'],
                labels=data_dict[split]['labels'],
                tokenizer=self.tokenizer,
                max_length=max_length
            )

            shuffle = (split == 'train')
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0
            )

        return loaders

    def fine_tune_model(self, data_loaders, method_name, epochs=3, learning_rate=2e-5, warmup_steps=100, save_model=True):
        """
        Fine-tune the model on the prepared data.
        """
        print(f"\n{'='*60}")
        print(f"FINE-TUNING MODEL: {method_name.upper()}")
        print(f"{'='*60}")

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)

        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(data_loaders['train']) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'epoch_details': []
        }

        # Training loop
        best_val_auc = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # Training phase
            model.train()
            total_train_loss = 0

            train_pbar = tqdm(data_loaders['train'], desc="Training")
            for batch in train_pbar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_train_loss / len(data_loaders['train'])

            # Validation phase
            val_loss, val_auc, val_metrics = self._evaluate_model(model, data_loaders['val'])

            # Save history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            epoch_detail = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_metrics': val_metrics
            }
            history['epoch_details'].append(epoch_detail)

            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f}")

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                print(f"✓ New best validation AUC: {val_auc:.4f}")

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model with validation AUC: {best_val_auc:.4f}")

        # Save model if requested
        if save_model:
            model_path = f"fine_tuned_{method_name}_model"
            model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            print(f"✓ Model saved to {model_path}")

        # Store training history
        self.training_history[method_name] = history

        return model, history

    def _evaluate_model(self, model, data_loader):
        """Evaluate model on given data loader."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                predictions = torch.sigmoid(outputs.logits)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Calculate AUC
        aucs = []
        for i in range(len(self.labels)):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                aucs.append(auc)

        mean_auc = np.mean(aucs) if aucs else 0.0

        metrics = {
            'mean_auc': mean_auc,
            'individual_aucs': {self.labels[i]: aucs[i] if i < len(aucs) else 0.0 
                               for i in range(len(self.labels))}
        }

        return avg_loss, mean_auc, metrics

    def evaluate_on_test_set(self, model, eval_texts, eval_labels, method_name, batch_size=16, max_length=256):
        """
        Evaluate fine-tuned model on test set.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING ON TEST SET: {method_name.upper()}")
        print(f"{'='*60}")

        # Create test dataset and loader
        test_dataset = ToxicCommentDataset(
            texts=eval_texts,
            labels=eval_labels,
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Evaluate
        test_loss, test_auc, test_metrics = self._evaluate_model(model, test_loader)

        # Get detailed predictions
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs.logits)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        # Analyze neutral performance
        neutral_analysis = self._analyze_neutral_performance(all_labels, all_predictions)

        results = {
            'method': method_name,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'test_metrics': test_metrics,
            'neutral_analysis': neutral_analysis,
            'predictions': all_predictions,
            'labels': all_labels
        }

        # Print results
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Mean AUC: {test_auc:.4f}")

        print(f"\nPer-label AUC:")
        for label, auc in test_metrics['individual_aucs'].items():
            print(f"  {label:15}: {auc:.4f}")

        print(f"\nNeutral Comment Analysis:")
        print(f"  Total neutral: {neutral_analysis['total_neutral']}")
        print(f"  Correctly classified: {neutral_analysis['correct_neutral']}")
        print(f"  Accuracy: {neutral_analysis['neutral_accuracy']:.4f}")
        print(f"  False positive rate: {neutral_analysis['fp_rate']:.4f}")

        return results

    def _analyze_neutral_performance(self, y_true, y_pred, threshold=0.5):
        """Analyze performance specifically on neutral comments."""
        y_pred_binary = (y_pred > threshold).astype(int)

        # Find neutral comments (no positive labels)
        neutral_mask = (y_true.sum(axis=1) == 0)
        total_neutral = neutral_mask.sum()

        if total_neutral == 0:
            return {
                'total_neutral': 0,
                'correct_neutral': 0,
                'neutral_accuracy': 0.0,
                'fp_rate': 0.0
            }

        # Check if neutral comments were correctly classified as neutral
        neutral_predictions = y_pred_binary[neutral_mask]
        correct_neutral = (neutral_predictions.sum(axis=1) == 0).sum()
        false_positives = total_neutral - correct_neutral

        return {
            'total_neutral': int(total_neutral),
            'correct_neutral': int(correct_neutral),
            'false_positives': int(false_positives),
            'neutral_accuracy': float(correct_neutral / total_neutral),
            'fp_rate': float(false_positives / total_neutral)
        }

    def compare_models(self, baseline_results, tagged_results):
        """
        Compare baseline and tagged model results.
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON: BASELINE vs SUBTLE_TOXICITY")
        print(f"{'='*80}")

        comparison = {}

        # Overall performance comparison
        baseline_auc = baseline_results['test_auc']
        tagged_auc = tagged_results['test_auc']
        auc_improvement = tagged_auc - baseline_auc
        auc_improvement_pct = (auc_improvement / baseline_auc) * 100 if baseline_auc > 0 else 0

        comparison['overall'] = {
            'baseline_auc': baseline_auc,
            'tagged_auc': tagged_auc,
            'improvement': auc_improvement,
            'improvement_pct': auc_improvement_pct
        }

        print(f"Overall Performance:")
        print(f"  Baseline AUC:     {baseline_auc:.4f}")
        print(f"  Tagged AUC:       {tagged_auc:.4f}")
        print(f"  Improvement:      {auc_improvement:+.4f} ({auc_improvement_pct:+.2f}%)")

        # Per-label comparison
        print(f"\nPer-Label Comparison:")
        print(f"{'Label':<15} {'Baseline':<10} {'Tagged':<10} {'Improvement':<12}")
        print("-" * 55)

        label_comparisons = {}
        for label in self.labels:
            baseline_label_auc = baseline_results['test_metrics']['individual_aucs'][label]
            tagged_label_auc = tagged_results['test_metrics']['individual_aucs'][label]

            if baseline_label_auc > 0:
                label_improvement = tagged_label_auc - baseline_label_auc
                label_improvement_pct = (label_improvement / baseline_label_auc) * 100
            else:
                label_improvement = 0
                label_improvement_pct = 0

            label_comparisons[label] = {
                'baseline': baseline_label_auc,
                'tagged': tagged_label_auc,
                'improvement': label_improvement,
                'improvement_pct': label_improvement_pct
            }

            print(f"{label:<15} {baseline_label_auc:<10.4f} {tagged_label_auc:<10.4f} "
                  f"{label_improvement:+.4f} ({label_improvement_pct:+.1f}%)")

        comparison['per_label'] = label_comparisons

        # Neutral performance comparison
        baseline_neutral = baseline_results['neutral_analysis']
        tagged_neutral = tagged_results['neutral_analysis']

        print(f"\nNeutral Comment Performance:")
        print(f"  Baseline accuracy: {baseline_neutral['neutral_accuracy']:.4f}")
        print(f"  Tagged accuracy:   {tagged_neutral['neutral_accuracy']:.4f}")
        print(f"  Baseline FP rate:  {baseline_neutral['fp_rate']:.4f}")
        print(f"  Tagged FP rate:    {tagged_neutral['fp_rate']:.4f}")

        comparison['neutral'] = {
            'baseline_accuracy': baseline_neutral['neutral_accuracy'],
            'tagged_accuracy': tagged_neutral['neutral_accuracy'],
            'baseline_fp_rate': baseline_neutral['fp_rate'],
            'tagged_fp_rate': tagged_neutral['fp_rate']
        }

        # Summary and recommendations
        print(f"\n{'='*80}")
        print("SUMMARY AND RECOMMENDATIONS")
        print(f"{'='*80}")

        if auc_improvement > 0.01:
            print("✅ SUBTLE_TOXICITY tagging shows meaningful improvement!")
            print(f"   Overall AUC improved by {auc_improvement:.4f} ({auc_improvement_pct:+.2f}%)")

            best_improvements = sorted(label_comparisons.items(), 
                                     key=lambda x: x[1]['improvement'], reverse=True)[:3]
            print("   Best improvements in:")
            for label, metrics in best_improvements:
                if metrics['improvement'] > 0.005:
                    print(f"     - {label}: +{metrics['improvement']:.4f} ({metrics['improvement_pct']:+.1f}%)")

        elif auc_improvement > -0.01:
            print("⚖️  SUBTLE_TOXICITY tagging shows neutral results")
            print("   Consider trying other tagging methods or combinations")

        else:
            print("❌ SUBTLE_TOXICITY tagging hurts performance")
            print("   Stick with baseline model or try different tagging approach")

        return comparison

    def plot_training_history(self, method_names=None):
        """Plot training history for comparison."""
        if method_names is None:
            method_names = list(self.training_history.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison', fontsize=16)

        # Training Loss
        axes[0, 0].set_title('Training Loss')
        for method in method_names:
            if method in self.training_history:
                history = self.training_history[method]
                axes[0, 0].plot(history['train_loss'], label=f'{method}', marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Validation Loss
        axes[0, 1].set_title('Validation Loss')
        for method in method_names:
            if method in self.training_history:
                history = self.training_history[method]
                axes[0, 1].plot(history['val_loss'], label=f'{method}', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Validation AUC
        axes[1, 0].set_title('Validation AUC')
        for method in method_names:
            if method in self.training_history:
                history = self.training_history[method]
                axes[1, 0].plot(history['val_auc'], label=f'{method}', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning curves comparison
        axes[1, 1].set_title('Training vs Validation Loss')
        for method in method_names:
            if method in self.training_history:
                history = self.training_history[method]
                epochs = range(1, len(history['train_loss']) + 1)
                axes[1, 1].plot(epochs, history['train_loss'], '--', label=f'{method} Train')
                axes[1, 1].plot(epochs, history['val_loss'], '-', label=f'{method} Val')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

        return fig

def run_fine_tuning_experiment(tagger, labels, device, X_train, y_train, X_val, y_val, 
                              eval_texts, eval_labels, epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Run the complete fine-tuning experiment using existing data splits.

    Args:
        tagger: ImprovedToxicCommentTagger instance
        labels: List of label names
        device: Device to use for training
        X_train: Training texts
        y_train: Training labels
        X_val: Validation texts  
        y_val: Validation labels
        eval_texts: Evaluation texts (balanced test set)
        eval_labels: Evaluation labels
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning

    Returns:
        Dictionary with all results and comparisons
    """
    print("🚀 STARTING FINE-TUNING EXPERIMENT")
    print("="*80)

    # Initialize fine-tuning manager
    ft_manager = FineTuningManager(tagger, labels, device)

    experiment_results = {}

    # Start MLflow experiment
    with mlflow.start_run(run_name="fine_tuning_baseline_vs_subtle_toxicity"):

        # 1. BASELINE EXPERIMENT
        print("\n🔵 PHASE 1: BASELINE MODEL FINE-TUNING")

        with mlflow.start_run(run_name="baseline_fine_tuning", nested=True):
            # Prepare baseline data
            baseline_data = ft_manager.prepare_training_data(X_train, y_train, X_val, y_val, method='baseline')
            baseline_loaders = ft_manager.create_data_loaders(baseline_data, batch_size)

            # Fine-tune baseline model
            baseline_model, baseline_history = ft_manager.fine_tune_model(
                baseline_loaders, 'baseline', epochs=epochs, learning_rate=learning_rate
            )

            # Evaluate baseline model
            baseline_results = ft_manager.evaluate_on_test_set(
                baseline_model, eval_texts, eval_labels, 'baseline'
            )

            # Log baseline metrics
            mlflow.log_metric("test_auc", baseline_results['test_auc'])
            mlflow.log_metric("test_loss", baseline_results['test_loss'])
            mlflow.log_param("method", "baseline")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)

        experiment_results['baseline'] = {
            'model': baseline_model,
            'history': baseline_history,
            'results': baseline_results
        }

        # 2. SUBTLE_TOXICITY EXPERIMENT
        print("\n🟢 PHASE 2: SUBTLE_TOXICITY MODEL FINE-TUNING")

        with mlflow.start_run(run_name="subtle_toxicity_fine_tuning", nested=True):
            # Prepare tagged data
            tagged_data = ft_manager.prepare_training_data(X_train, y_train, X_val, y_val, method='subtle_toxicity')
            tagged_loaders = ft_manager.create_data_loaders(tagged_data, batch_size)

            # Fine-tune tagged model
            tagged_model, tagged_history = ft_manager.fine_tune_model(
                tagged_loaders, 'subtle_toxicity', epochs=epochs, learning_rate=learning_rate
            )

            # Evaluate tagged model (apply tagging to eval texts too)
            eval_texts_tagged = tagger.apply_tagging_method(eval_texts, 'subtle_toxicity')
            tagged_results = ft_manager.evaluate_on_test_set(
                tagged_model, eval_texts_tagged, eval_labels, 'subtle_toxicity'
            )

            # Log tagged metrics
            mlflow.log_metric("test_auc", tagged_results['test_auc'])
            mlflow.log_metric("test_loss", tagged_results['test_loss'])
            mlflow.log_param("method", "subtle_toxicity")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)

        experiment_results['subtle_toxicity'] = {
            'model': tagged_model,
            'history': tagged_history,
            'results': tagged_results
        }

        # 3. COMPARISON AND ANALYSIS
        print("\n📊 PHASE 3: COMPREHENSIVE COMPARISON")
        comparison = ft_manager.compare_models(baseline_results, tagged_results)
        experiment_results['comparison'] = comparison

        # Log comparison metrics
        mlflow.log_metric("auc_improvement", comparison['overall']['improvement'])
        mlflow.log_metric("auc_improvement_pct", comparison['overall']['improvement_pct'])

        # 4. VISUALIZATIONS
        print("\n📈 PHASE 4: GENERATING VISUALIZATIONS")
        training_plot = ft_manager.plot_training_history(['baseline', 'subtle_toxicity'])
        experiment_results['training_plot'] = training_plot

        # Save plot
        training_plot.savefig('training_history_comparison.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('training_history_comparison.png')

    print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)

    return experiment_results, ft_manager

# Example usage with existing variables
def example_integration():
    """
    Example of how to integrate with existing code setup.
    """
    example_code = '''
    # Using your existing variables:
    # - tagger (ImprovedToxicCommentTagger instance)
    # - labels (list of label names)
    # - device (torch device)
    # - X_train, y_train (training data)
    # - X_val, y_val (validation data)  
    # - eval_texts, eval_labels (balanced evaluation set)

    # Run the fine-tuning experiment
    results, ft_manager = run_fine_tuning_experiment(
        tagger=tagger,
        labels=labels,
        device=device,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )

    # Access results
    baseline_auc = results['baseline']['results']['test_auc']
    tagged_auc = results['subtle_toxicity']['results']['test_auc']
    improvement = results['comparison']['overall']['improvement']

    print(f"Baseline AUC: {baseline_auc:.4f}")
    print(f"Tagged AUC: {tagged_auc:.4f}")
    print(f"Improvement: {improvement:+.4f}")

    # Use the best model for future predictions
    if improvement > 0:
        best_model = results['subtle_toxicity']['model']
        print("Using SUBTLE_TOXICITY model (better performance)")
    else:
        best_model = results['baseline']['model']
        print("Using baseline model (better performance)")
    '''

    print("Integration Example:")
    print("="*50)
    print(example_code)

if __name__ == "__main__":
    example_integration()
