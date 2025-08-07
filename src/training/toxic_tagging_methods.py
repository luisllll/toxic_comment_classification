

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ToxicCommentTagger:
    """
    Advanced tagging strategies for toxic comment classification enhancement.
    """
    
    def __init__(self, base_model_name: str = "unitary/toxic-bert", device: str = None):
        self.base_model_name = base_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Initialized ToxicCommentTagger with {base_model_name} on {self.device}")
    
    def load_model(self):
        """Load the base model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)
        model.eval()
        return model
    
    # ============================================================================
    # METHOD 1: EXPLICIT TOXICITY MARKERS TAGGING
    # ============================================================================
    
    def method_1_explicit_markers(self, text: str) -> str:
        """
        Method 1: Add explicit toxicity type markers to text.
        
        Strategy: Prepend text with detected toxicity indicators to help the model
        focus on specific patterns.
        
        Example: "[THREAT][INSULT] you are stupid and I will hurt you"
        """
        markers = []
        text_lower = text.lower()
        
        # Define toxicity patterns
        toxicity_patterns = {
            'THREAT': [
                r'\b(kill|murder|hurt|harm|attack|destroy|beat)\b.*\b(you|u)\b',
                r'\bi will\b.*\b(kill|hurt|harm|get|find)\b',
                r'\byou.*(die|dead|killed)\b',
                r'\bwatch out\b', r'\byou.re.dead\b'
            ],
            'SEVERE_TOXIC': [
                r'\b(fucking|goddamn|motherfucker|asshole|bitch|cunt|shit|damn)\b.*\b(fucking|goddamn|motherfucker|asshole|bitch|cunt|shit|damn)\b',
                r'\bfuck.*fuck\b', r'\bshit.*shit\b',
                r'\b(kill yourself|kys)\b'
            ],
            'OBSCENE': [
                r'\b(fuck|fucking|shit|damn|ass|bitch|piss|crap)\b',
                r'\b(penis|vagina|dick|cock|pussy)\b',
                r'\bsex\b', r'\bporn\b'
            ],
            'INSULT': [
                r'\b(stupid|idiot|moron|dumb|retard|loser|pathetic|worthless)\b',
                r'\byou are\b.*(stupid|dumb|ugly|fat|worthless)',
                r'\b(go away|shut up|stfu)\b'
            ],
            'IDENTITY_HATE': [
                r'\b(gay|homo|fag|lesbian|trans|black|white|jew|muslim|christian)\b.*\b(hate|bad|evil|wrong|disgusting)\b',
                r'\b(nigger|faggot|kike|spic|chink)\b',
                r'\bpeople like you\b', r'\byour kind\b'
            ]
        }
        
        # Check each pattern
        for toxicity_type, patterns in toxicity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if toxicity_type not in markers:
                        markers.append(toxicity_type)
                    break
        
        # Add general toxic marker if any toxicity detected
        if markers and 'TOXIC' not in markers:
            markers.insert(0, 'TOXIC')
        
        # Format the tagged text
        if markers:
            marker_string = ''.join([f'[{marker}]' for marker in markers])
            return f"{marker_string} {text}"
        else:
            return f"[NEUTRAL] {text}"
    
    # ============================================================================
    # METHOD 2: CONTEXTUAL INTENSITY TAGGING
    # ============================================================================
    
    def method_2_intensity_tagging(self, text: str) -> str:
        """
        Method 2: Add intensity level tags based on linguistic features.
        
        Strategy: Analyze linguistic intensity markers (caps, repetition, 
        punctuation) to help model understand toxicity severity.
        
        Example: "[INTENSITY:HIGH][CAPS:HEAVY] YOU ARE SO STUPID!!!"
        """
        intensity_score = 0
        caps_ratio = 0
        tags = []
        
        # Calculate caps ratio
        if len(text) > 0:
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / len(text)
        
        # Caps intensity
        if caps_ratio > 0.7:
            tags.append("CAPS:HEAVY")
            intensity_score += 3
        elif caps_ratio > 0.4:
            tags.append("CAPS:MODERATE")
            intensity_score += 2
        elif caps_ratio > 0.2:
            tags.append("CAPS:LIGHT")
            intensity_score += 1
        
        # Repetition patterns
        repetition_patterns = [
            r'(.)\1{2,}',  # Character repetition (aaa, !!! )
            r'\b(\w+)\s+\1\b',  # Word repetition
            r'[!]{2,}', r'[?]{2,}', r'[.]{3,}'  # Punctuation repetition
        ]
        
        repetition_count = 0
        for pattern in repetition_patterns:
            repetition_count += len(re.findall(pattern, text))
        
        if repetition_count >= 3:
            tags.append("REPEAT:HEAVY")
            intensity_score += 3
        elif repetition_count >= 2:
            tags.append("REPEAT:MODERATE")
            intensity_score += 2
        elif repetition_count >= 1:
            tags.append("REPEAT:LIGHT")
            intensity_score += 1
        
        # Profanity intensity
        strong_profanity = len(re.findall(r'\b(fuck|shit|damn|ass|bitch)\b', text.lower()))
        if strong_profanity >= 3:
            tags.append("PROFANITY:HEAVY")
            intensity_score += 3
        elif strong_profanity >= 2:
            tags.append("PROFANITY:MODERATE")
            intensity_score += 2
        elif strong_profanity >= 1:
            tags.append("PROFANITY:LIGHT")
            intensity_score += 1
        
        # Overall intensity
        if intensity_score >= 6:
            tags.insert(0, "INTENSITY:HIGH")
        elif intensity_score >= 3:
            tags.insert(0, "INTENSITY:MODERATE")
        elif intensity_score >= 1:
            tags.insert(0, "INTENSITY:LOW")
        else:
            tags.insert(0, "INTENSITY:NEUTRAL")
        
        # Format tagged text
        if tags:
            tag_string = ''.join([f'[{tag}]' for tag in tags])
            return f"{tag_string} {text}"
        else:
            return f"[INTENSITY:NEUTRAL] {text}"
    
    # ============================================================================
    # METHOD 3: TARGET-ORIENTED TAGGING
    # ============================================================================
    
    def method_3_target_tagging(self, text: str) -> str:
        """
        Method 3: Tag based on attack targets and directions.
        
        Strategy: Identify who/what is being targeted to help model understand
        the nature of the toxic behavior.
        
        Example: "[TARGET:PERSON][DIRECTION:DIRECT] you are stupid"
        """
        tags = []
        text_lower = text.lower()
        
        # Target identification
        target_patterns = {
            'PERSON': [
                r'\b(you|u|your|yours)\b', r'\b(he|she|him|her|they|them)\b',
                r'\b(guy|girl|man|woman|boy|girl|person|people)\b'
            ],
            'GROUP': [
                r'\b(people like you|your kind|you guys|you all|y\'all)\b',
                r'\b(everyone|everybody|nobody|anyone)\b',
                r'\b(women|men|girls|boys|kids|children)\b'
            ],
            'IDENTITY': [
                r'\b(gay|lesbian|trans|black|white|asian|hispanic|muslim|christian|jew)\b.*\b(people|person|community)\b',
                r'\b(liberals|conservatives|democrats|republicans)\b'
            ]
        }
        
        # Direction identification
        direction_patterns = {
            'DIRECT': [r'\byou\b', r'\byour\b', r'\bu\b'],
            'INDIRECT': [r'\bpeople like\b', r'\bthose who\b', r'\banyone who\b'],
            'GENERAL': [r'\beveryone\b', r'\bpeople\b', r'\bworld\b']
        }
        
        # Check targets
        for target_type, patterns in target_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"TARGET:{target_type}")
                    break
        
        # Check directions
        for direction_type, patterns in direction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"DIRECTION:{direction_type}")
                    break
        
        # Attack type
        attack_patterns = {
            'PERSONAL': [r'\byou are\b', r'\byou look\b', r'\byou seem\b'],
            'BEHAVIORAL': [r'\byou do\b', r'\byou always\b', r'\byou never\b'],
            'THREATENING': [r'\bi will\b', r'\bwatch out\b', r'\byou better\b'],
            'DISMISSIVE': [r'\bshut up\b', r'\bgo away\b', r'\bget lost\b']
        }
        
        for attack_type, patterns in attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"ATTACK:{attack_type}")
                    break
        
        # Format tagged text
        if not tags:
            tags = ["TARGET:NONE"]
        
        tag_string = ''.join([f'[{tag}]' for tag in tags])
        return f"{tag_string} {text}"
    
    # ============================================================================
    # METHOD 4: LINGUISTIC FEATURE TAGGING
    # ============================================================================
    
    def method_4_linguistic_features(self, text: str) -> str:
        """
        Method 4: Add linguistic and structural feature tags.
        
        Strategy: Tag based on sentence structure, length, and linguistic
        patterns that correlate with toxicity.
        
        Example: "[LENGTH:SHORT][QUESTION][IMPERATIVE] why are you so dumb?"
        """
        tags = []
        
        # Length analysis
        word_count = len(text.split())
        if word_count <= 5:
            tags.append("LENGTH:SHORT")
        elif word_count <= 15:
            tags.append("LENGTH:MEDIUM")
        else:
            tags.append("LENGTH:LONG")
        
        # Sentence type analysis
        if '?' in text:
            tags.append("QUESTION")
        if '!' in text:
            tags.append("EXCLAMATION")
        
        # Imperative detection (commands)
        imperative_patterns = [
            r'^\s*(go|get|shut|stop|don\'t|do|be|make)\b',
            r'^\s*(you\s+)?(should|must|need to|have to)\b'
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, text.lower()):
                tags.append("IMPERATIVE")
                break
        
        # Negation analysis
        negation_count = len(re.findall(r'\b(not|no|never|nothing|nobody|nowhere|neither|none|don\'t|doesn\'t|didn\'t|won\'t|can\'t|shouldn\'t|wouldn\'t)\b', text.lower()))
        if negation_count >= 2:
            tags.append("NEGATION:HEAVY")
        elif negation_count >= 1:
            tags.append("NEGATION:PRESENT")
        
        # Comparison patterns (often used in insults)
        comparison_patterns = [
            r'\b(more|less|better|worse|smarter|dumber|uglier|prettier)\b.*\bthan\b',
            r'\bas\s+\w+\s+as\b',
            r'\blike\s+\w+\b'
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, text.lower()):
                tags.append("COMPARISON")
                break
        
        # Format tagged text
        tag_string = ''.join([f'[{tag}]' for tag in tags])
        return f"{tag_string} {text}"
    
    # ============================================================================
    # METHOD 5: SEMANTIC CONTEXT TAGGING
    # ============================================================================
    
    def method_5_semantic_context(self, text: str) -> str:
        """
        Method 5: Add semantic context tags based on topic and domain.
        
        Strategy: Identify the context/domain of the conversation to help
        model understand situational toxicity.
        
        Example: "[CONTEXT:POLITICAL][EMOTION:ANGER] politicians are all corrupt!"
        """
        tags = []
        text_lower = text.lower()
        
        # Context/Domain detection
        context_patterns = {
            'POLITICAL': [
                r'\b(politician|politics|government|election|vote|democrat|republican|liberal|conservative|president|congress)\b'
            ],
            'PERSONAL': [
                r'\b(family|mother|father|wife|husband|girlfriend|boyfriend|friend|relationship)\b'
            ],
            'APPEARANCE': [
                r'\b(ugly|beautiful|fat|thin|tall|short|look|face|body|hair|clothes)\b'
            ],
            'PERFORMANCE': [
                r'\b(work|job|school|grade|test|performance|skill|talent|ability|smart|stupid)\b'
            ],
            'IDENTITY': [
                r'\b(race|religion|sexuality|gender|culture|background|country|ethnicity)\b'
            ],
            'GAMING': [
                r'\b(game|player|team|match|win|lose|noob|pro|skill|rank)\b'
            ]
        }
        
        # Emotional context
        emotion_patterns = {
            'ANGER': [
                r'\b(angry|mad|pissed|furious|hate|rage|annoyed|irritated)\b',
                r'\b(what the hell|wtf|damn|goddamn)\b'
            ],
            'FRUSTRATION': [
                r'\b(frustrated|tired|sick|done|enough|can\'t stand)\b'
            ],
            'CONTEMPT': [
                r'\b(disgusting|pathetic|worthless|useless|waste|trash)\b'
            ],
            'DISMISSIVE': [
                r'\b(whatever|who cares|don\'t care|shut up|ignore)\b'
            ]
        }
        
        # Check contexts
        for context_type, patterns in context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"CONTEXT:{context_type}")
                    break
        
        # Check emotions
        for emotion_type, patterns in emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"EMOTION:{emotion_type}")
                    break
        
        # Social dynamics
        social_patterns = {
            'SUPERIORITY': [r'\bi am better\b', r'\byou are worse\b', r'\bi know more\b'],
            'EXCLUSION': [r'\bget out\b', r'\bdon\'t belong\b', r'\bnot welcome\b'],
            'DOMINANCE': [r'\bi will show you\b', r'\byou will\b', r'\bmake you\b']
        }
        
        for social_type, patterns in social_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(f"SOCIAL:{social_type}")
                    break
        
        # Default context if none found
        if not any(tag.startswith('CONTEXT:') for tag in tags):
            tags.append("CONTEXT:GENERAL")
        
        # Format tagged text
        tag_string = ''.join([f'[{tag}]' for tag in tags])
        return f"{tag_string} {text}"
    
    # ============================================================================
    # EVALUATION AND COMPARISON FRAMEWORK
    # ============================================================================
    
    def apply_tagging_method(self, texts: List[str], method: str) -> List[str]:
        """Apply a specific tagging method to a list of texts."""
        method_map = {
            'explicit_markers': self.method_1_explicit_markers,
            'intensity_tagging': self.method_2_intensity_tagging,
            'target_tagging': self.method_3_target_tagging,
            'linguistic_features': self.method_4_linguistic_features,
            'semantic_context': self.method_5_semantic_context,
            'baseline': lambda x: x  # No tagging
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}")
        
        tagging_func = method_map[method]
        return [tagging_func(text) for text in tqdm(texts, desc=f"Applying {method}")]
    
    def evaluate_tagging_method(self, 
                               texts: List[str], 
                               labels: np.ndarray,
                               method: str,
                               max_length: int = 256,
                               batch_size: int = 16) -> Dict:
        """
        Evaluate a tagging method by comparing performance with baseline.
        
        Returns metrics including AUC scores and neutral performance.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING TAGGING METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        # Apply tagging method
        tagged_texts = self.apply_tagging_method(texts, method)
        
        # Show examples
        print(f"\nExample transformations:")
        for i in range(min(3, len(texts))):
            print(f"Original: {texts[i]}")
            print(f"Tagged:   {tagged_texts[i]}")
            print()
        
        # Load model
        model = self.load_model()
        
        # Evaluate
        predictions = []
        
        for i in tqdm(range(0, len(tagged_texts), batch_size), desc="Evaluating"):
            batch_texts = tagged_texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate metrics
        results = self._calculate_metrics(labels, predictions, method)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, method: str) -> Dict:
        """Calculate comprehensive metrics for evaluation."""
        results = {'method': method}
        
        # Per-label AUC
        label_aucs = []
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                label_aucs.append(auc)
                results[f'auc_{label}'] = auc
                print(f"  {label}: {auc:.4f}")
            else:
                label_aucs.append(0.0)
                results[f'auc_{label}'] = 0.0
                print(f"  {label}: No samples")
        
        # Mean AUC
        mean_auc = np.mean([auc for auc in label_aucs if auc > 0])
        results['mean_auc'] = mean_auc
        print(f"  Mean AUC: {mean_auc:.4f}")
        
        # Neutral performance
        neutral_metrics = self._evaluate_neutral_performance(y_true, y_pred)
        results.update(neutral_metrics)
        
        print(f"  Neutral accuracy: {neutral_metrics['neutral_accuracy']:.4f}")
        print(f"  Neutral FP rate: {neutral_metrics['neutral_fp_rate']:.4f}")
        
        return results
    
    def _evaluate_neutral_performance(self, y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict:
        """Evaluate performance on neutral comments."""
        y_pred_bin = (y_pred_probs > threshold).astype(int)
        neutral_mask = (y_true.sum(axis=1) == 0)
        neutral_total = neutral_mask.sum()
        
        if neutral_total == 0:
            return {
                'neutral_total': 0,
                'neutral_correct': 0,
                'neutral_accuracy': 0.0,
                'neutral_fp_rate': 0.0
            }
        
        neutral_pred_mask = (y_pred_bin[neutral_mask].sum(axis=1) == 0)
        neutral_correct = neutral_pred_mask.sum()
        neutral_fp = neutral_total - neutral_correct
        
        return {
            'neutral_total': int(neutral_total),
            'neutral_correct': int(neutral_correct),
            'neutral_fp': int(neutral_fp),
            'neutral_accuracy': float(neutral_correct / neutral_total),
            'neutral_fp_rate': float(neutral_fp / neutral_total)
        }
    
    def compare_all_methods(self, 
                           texts: List[str], 
                           labels: np.ndarray,
                           methods: List[str] = None) -> pd.DataFrame:
        """
        Compare all tagging methods and return results DataFrame.
        
        Args:
            texts: List of comment texts
            labels: Ground truth labels (n_samples, n_labels)
            methods: List of methods to compare (default: all methods)
        
        Returns:
            DataFrame with comparison results
        """
        if methods is None:
            methods = [
                'baseline',
                'explicit_markers', 
                'intensity_tagging',
                'target_tagging',
                'linguistic_features',
                'semantic_context'
            ]
        
        results = []
        
        with mlflow.start_run(run_name="tagging_methods_comparison"):
            for method in methods:
                print(f"\n{'='*80}")
                print(f"EVALUATING METHOD: {method.upper()}")
                print(f"{'='*80}")
                
                try:
                    # Evaluate method
                    method_results = self.evaluate_tagging_method(texts, labels, method)
                    results.append(method_results)
                    
                    # Log to MLflow
                    with mlflow.start_run(run_name=f"method_{method}", nested=True):
                        for metric, value in method_results.items():
                            if isinstance(value, (int, float)) and metric != 'method':
                                mlflow.log_metric(metric, value)
                        mlflow.log_param('tagging_method', method)
                
                except Exception as e:
                    print(f"❌ Error evaluating {method}: {str(e)}")
                    continue
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by mean AUC
        results_df = results_df.sort_values('mean_auc', ascending=False)
        
        print(f"\n{'='*100}")
        print("TAGGING METHODS COMPARISON RESULTS")
        print(f"{'='*100}")
        print(f"{'Method':<20} {'Mean AUC':<12} {'Neutral Acc':<12} {'Neutral FP':<12} {'Best Labels':<30}")
        print("-" * 100)
        
        for _, row in results_df.iterrows():
            # Find best performing labels for this method
            label_aucs = [row[f'auc_{label}'] for label in self.labels]
            best_labels = [self.labels[i] for i, auc in enumerate(label_aucs) if auc == max(label_aucs)]
            best_labels_str = ', '.join(best_labels[:2]) + ('...' if len(best_labels) > 2 else '')
            
            print(f"{row['method']:<20} {row['mean_auc']:<12.4f} {row['neutral_accuracy']:<12.4f} "
                  f"{row['neutral_fp_rate']:<12.4f} {best_labels_str:<30}")
        
        return results_df


# ============================================================================
# USAGE EXAMPLE AND TESTING FUNCTIONS
# ============================================================================

def demo_tagging_methods():
    """Demonstrate all tagging methods with example texts."""
    tagger = ToxicCommentTagger()
    
    example_texts = [
        "you are so stupid and ugly",
        "I HATE YOU SO MUCH!!!",
        "shut up you idiot, nobody likes you",
        "politicians are all corrupt and evil",
        "kill yourself you worthless piece of trash",
        "this is a normal comment about the weather"
    ]
    
    methods = [
        'explicit_markers',
        'intensity_tagging', 
        'target_tagging',
        'linguistic_features',
        'semantic_context'
    ]
    
    print("TAGGING METHODS DEMONSTRATION")
    print("=" * 80)
    
    for text in example_texts:
        print(f"\nOriginal: {text}")
        print("-" * 50)
        
        for method in methods:
            try:
                if method == 'explicit_markers':
                    tagged = tagger.method_1_explicit_markers(text)
                elif method == 'intensity_tagging':
                    tagged = tagger.method_2_intensity_tagging(text)
                elif method == 'target_tagging':
                    tagged = tagger.method_3_target_tagging(text)
                elif method == 'linguistic_features':
                    tagged = tagger.method_4_linguistic_features(text)
                elif method == 'semantic_context':
                    tagged = tagger.method_5_semantic_context(text)
                
                print(f"{method:20}: {tagged}")
                
            except Exception as e:
                print(f"{method:20}: Error - {str(e)}")


if __name__ == "__main__":
    demo_tagging_methods()



"""
Improved Toxic Comment Tagging Methods
=====================================

This module implements improved tagging strategies based on analysis showing that
heavy tagging hurts performance (negative Cohen's d values). 

Key improvements:
1. Minimal, high-confidence tagging only
2. Focus on edge cases toxic-bert might miss
3. Avoid redundant information
4. Preserve model's existing strengths

Author: Data Science Team  
Version: 2.0 (Improved)
Date: August 2025
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ImprovedToxicCommentTagger:
    """
    Improved tagging strategies that complement toxic-bert without overwhelming it.
    
    Philosophy:
    - Less is more: Only tag when highly confident
    - Focus on edge cases the model might miss
    - Preserve the model's existing strengths
    - Use semantic understanding over pattern matching
    """
    
    def __init__(self, base_model_name: str = "unitary/toxic-bert", device: str = None):
        self.base_model_name = base_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize high-confidence pattern databases
        self._init_pattern_databases()
        
        print(f"Initialized ImprovedToxicCommentTagger with {base_model_name} on {self.device}")
    
    def _init_pattern_databases(self):
        """Initialize high-confidence pattern databases for different toxicity types."""
        
        # High-confidence explicit threats (very specific patterns)
        self.explicit_threats = [
            r'\bi will kill you\b',
            r'\bi am going to (kill|murder|hurt) you\b',
            r'\bi hope you die\b',
            r'\byou (should|need to) die\b',
            r'\bkill yourself\b',
            r'\bi will find you and\b',
            r'\byou are dead\b.*\bmeat\b',
            r'\bi will destroy you\b'
        ]
        
        # High-confidence identity attacks (slurs and explicit hate)
        self.identity_slurs = [
            r'\b(nigger|faggot|kike|spic|chink|tranny)\b',
            r'\b(raghead|towelhead|wetback)\b'
        ]
        
        # High-confidence severe profanity combinations
        self.severe_profanity = [
            r'\bfucking\s+(piece\s+of\s+)?shit\b',
            r'\bmotherfucker\b',
            r'\bgoddamn\s+(fucking\s+)?\w+\b',
            r'\bcunt\b',
            r'\bfuck\s+you\s+(you\s+)?\w+\b'
        ]
        
        # Subtle toxicity patterns toxic-bert might miss
        self.subtle_toxicity = {
            'sarcastic': [
                r'\b(oh\s+)?(wow|great|wonderful|brilliant|genius)\b.*\b(idiot|moron|stupid)\b',
                r'\bcongratulations\b.*\b(being|for)\s+(stupid|dumb|an idiot)\b',
                r'\bwell done\b.*\b(einstein|genius)\b',
                r'\bgood job\b.*\b(idiot|moron)\b'
            ],
            'passive_aggressive': [
                r'\bno offense\s+(but|however|although)\b',
                r'\bi.m not\s+(racist|sexist|homophobic)\s+but\b',
                r'\bwith all due respect\b.*\bbut\b',
                r'\bjust saying\b.*\bbut\b',
                r'\bbless your heart\b'
            ],
            'coded_language': [
                r'\burban\s+(youth|people|culture)\b',
                r'\binner\s+city\s+(people|culture|problems)\b',
                r'\bthose\s+people\b',
                r'\bcertain\s+people\b',
                r'\byou\s+people\b',
                r'\bwelfare\s+queens?\b'
            ],
            'conditional_threats': [
                r'\bif\s+you\s+(don.?t|do\s+not)\b.*\bi\s+(will|ll)\b',
                r'\byou\s+better\b.*\bor\s+(else|i.ll|i\s+will)\b',
                r'\bunless\s+you\b.*\bi.ll\b',
                r'\bone\s+more\s+(word|time)\b.*\band\b'
            ]
        }
    
    def load_model(self):
        """Load the base model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)
        model.eval()
        return model
    
    # ============================================================================
    # IMPROVED METHOD 1: HIGH-CONFIDENCE EXPLICIT TOXICITY
    # ============================================================================
    
    def method_high_confidence_explicit(self, text: str) -> str:
        """
        Method 1: Only tag when we have very high confidence about explicit toxicity.
        
        Strategy: Use precise patterns to identify toxicity that should definitely
        be flagged, avoiding false positives that confuse the model.
        
        Conservative approach: Better to miss some than to add noise.
        """
        tags = []
        text_lower = text.lower()
        
        # Check for explicit threats (very high confidence patterns)
        for pattern in self.explicit_threats:
            if re.search(pattern, text_lower):
                tags.append("EXPLICIT_THREAT")
                break
        
        # Check for identity slurs (unambiguous hate speech)
        for pattern in self.identity_slurs:
            if re.search(pattern, text_lower):
                tags.append("HATE_SLUR")
                break
        
        # Check for severe profanity combinations
        for pattern in self.severe_profanity:
            if re.search(pattern, text_lower):
                tags.append("SEVERE_PROFANITY")
                break
        
        # Only add tags if we found something unambiguous
        if tags:
            # Limit to most serious tag to avoid overwhelming
            priority_order = ["HATE_SLUR", "EXPLICIT_THREAT", "SEVERE_PROFANITY"]
            highest_priority = next((tag for tag in priority_order if tag in tags), tags[0])
            return f"[{highest_priority}] {text}"
        else:
            return text
    
    # ============================================================================
    # IMPROVED METHOD 2: SUBTLE TOXICITY DETECTION
    # ============================================================================
    
    def method_subtle_toxicity(self, text: str) -> str:
        """
        Method 2: Detect subtle forms of toxicity toxic-bert might miss.
        
        Strategy: Focus on implicit toxicity, sarcasm, coded language, and
        passive-aggressive patterns that require more context to understand.
        """
        detected_patterns = []
        text_lower = text.lower()
        
        # Check each category of subtle toxicity
        for category, patterns in self.subtle_toxicity.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_patterns.append(category.upper())
                    break  # Only one tag per category
        
        # Only tag if we found subtle toxicity patterns
        if detected_patterns:
            # Use only the first detected pattern to avoid clutter
            tag = detected_patterns[0]
            return f"[SUBTLE_{tag}] {text}"
        else:
            return text
    
    # ============================================================================
    # IMPROVED METHOD 3: CONTEXTUAL AMPLIFICATION
    # ============================================================================
    
    def method_contextual_amplification(self, text: str) -> str:
        """
        Method 3: Tag only when there are strong amplification signals.
        
        Strategy: Help the model recognize when toxic content is being
        amplified through repetition, caps, or multiple intensifiers.
        """
        amplification_score = 0
        text_lower = text.lower()
        
        # Strong repetition patterns
        if re.search(r'(.)\1{3,}', text):  # 4+ repeated characters
            amplification_score += 2
        elif re.search(r'(.)\1{2}', text):  # 3 repeated characters
            amplification_score += 1
        
        # Word repetition
        if re.search(r'\b(\w+)\s+\1\b', text_lower):
            amplification_score += 1
        
        # Multiple punctuation
        exclamation_count = len(re.findall(r'!', text))
        question_count = len(re.findall(r'\?', text))
        if exclamation_count >= 3 or question_count >= 3:
            amplification_score += 2
        elif exclamation_count >= 2 or question_count >= 2:
            amplification_score += 1
        
        # Caps analysis (be very conservative)
        if len(text) > 10:  # Only for longer texts
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.8:  # Very high caps
                amplification_score += 2
            elif caps_ratio > 0.6:  # High caps
                amplification_score += 1
        
        # Multiple intensifiers
        intensifiers = ['so', 'very', 'really', 'extremely', 'fucking', 'goddamn']
        intensifier_count = sum(1 for word in intensifiers if word in text_lower)
        if intensifier_count >= 2:
            amplification_score += 1
        
        # Only tag if amplification is strong AND there might be toxicity
        toxic_indicators = ['hate', 'stupid', 'idiot', 'moron', 'kill', 'die', 'fuck', 'shit']
        has_toxic_content = any(word in text_lower for word in toxic_indicators)
        
        if amplification_score >= 3 and has_toxic_content:
            return f"[AMPLIFIED] {text}"
        elif amplification_score >= 4:  # Very strong amplification even without obvious toxicity
            return f"[AMPLIFIED] {text}"
        else:
            return text
    
    # ============================================================================
    # IMPROVED METHOD 4: THREAT ESCALATION DETECTION
    # ============================================================================
    
    def method_threat_escalation(self, text: str) -> str:
        """
        Method 4: Detect escalating language that builds toward threats.
        
        Strategy: Identify patterns where someone is building up anger
        or moving toward threatening language.
        """
        escalation_indicators = []
        text_lower = text.lower()
        
        # Frustration buildup
        frustration_patterns = [
            r'\bi am (so|getting|really)\s+(tired|sick|fed up|mad|angry)\s+(of|with|at)\b',
            r'\bi (can.?t|cannot)\s+(take|stand|believe)\s+(this|it|you)\s+(anymore|any more)\b',
            r'\bthat.?s\s+it\b',
            r'\bi.?ve\s+had\s+(it|enough)\b'
        ]
        
        for pattern in frustration_patterns:
            if re.search(pattern, text_lower):
                escalation_indicators.append("FRUSTRATION")
                break
        
        # Warning language
        warning_patterns = [
            r'\byou\s+(better|need to|should)\s+(watch|be careful|stop)\b',
            r'\blast\s+(warning|chance|time)\b',
            r'\bdon.?t\s+(test|push|try)\s+me\b',
            r'\byou.?re\s+(asking for|going to get)\b'
        ]
        
        for pattern in warning_patterns:
            if re.search(pattern, text_lower):
                escalation_indicators.append("WARNING")
                break
        
        # Consequence language
        consequence_patterns = [
            r'\byou.?ll\s+(regret|be sorry|pay)\b',
            r'\bi.?ll\s+(make sure|show you|teach you)\b',
            r'\bwait\s+(and see|until)\b',
            r'\byou.?re\s+going to\s+(regret|be sorry)\b'
        ]
        
        for pattern in consequence_patterns:
            if re.search(pattern, text_lower):
                escalation_indicators.append("CONSEQUENCE")
                break
        
        # Only tag if we found escalation patterns
        if escalation_indicators:
            # Use the most serious escalation type
            priority = ["CONSEQUENCE", "WARNING", "FRUSTRATION"]
            tag = next((ind for ind in priority if ind in escalation_indicators), escalation_indicators[0])
            return f"[ESCALATING_{tag}] {text}"
        else:
            return text
    
    # ============================================================================
    # IMPROVED METHOD 5: PRECISION TARGETING
    # ============================================================================
    
    def method_precision_targeting(self, text: str) -> str:
        """
        Method 5: Identify targeting patterns that indicate directed attacks.
        
        Strategy: Focus on language that targets specific individuals or groups
        in ways that toxic-bert might not fully capture the directed nature.
        """
        targeting_score = 0
        targeting_type = None
        text_lower = text.lower()
        
        # Direct personal attacks
        direct_patterns = [
            r'\byou\s+(personally|specifically|in particular)\b',
            r'\bi.m\s+talking\s+to\s+you\b',
            r'\byou\s+know\s+who\s+you\s+are\b',
            r'\byou\s+especially\b'
        ]
        
        for pattern in direct_patterns:
            if re.search(pattern, text_lower):
                targeting_score += 2
                targeting_type = "PERSONAL"
                break
        
        # Group targeting
        group_patterns = [
            r'\ball\s+(of\s+)?you\s+(people|guys|folks)\b',
            r'\bpeople\s+like\s+you\b',
            r'\byour\s+(kind|type|sort)\b',
            r'\byou\s+and\s+your\s+(friends|family|people)\b'
        ]
        
        for pattern in group_patterns:
            if re.search(pattern, text_lower):
                targeting_score += 1
                if not targeting_type:
                    targeting_type = "GROUP"
        
        # Exclusionary language
        exclusion_patterns = [
            r'\bget\s+out\s+(of here|of this)\b',
            r'\byou\s+(don.?t belong|aren.?t welcome)\b',
            r'\bgo\s+(back|home)\s+(to|where you came from)\b',
            r'\bleave\s+(us|this place)\s+alone\b'
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                targeting_score += 2
                targeting_type = "EXCLUSION"
                break
        
        # Identity-based targeting
        identity_targeting = [
            r'\bas\s+a\s+(woman|man|gay|straight|black|white|muslim|christian|jew)\b',
            r'\byou\s+(women|men|gays|blacks|whites|muslims|christians|jews)\b',
            r'\bbecause\s+you.?re\s+(a\s+)?(woman|man|gay|black|white|muslim|jew)\b'
        ]
        
        for pattern in identity_targeting:
            if re.search(pattern, text_lower):
                targeting_score += 2
                targeting_type = "IDENTITY"
                break
        
        # Only tag if targeting score is significant
        if targeting_score >= 2 and targeting_type:
            return f"[TARGET_{targeting_type}] {text}"
        else:
            return text
    
    # ============================================================================
    # EVALUATION AND COMPARISON FRAMEWORK
    # ============================================================================
    
    def apply_tagging_method(self, texts: List[str], method: str) -> List[str]:
        """Apply a specific improved tagging method to a list of texts."""
        method_map = {
            'high_confidence_explicit': self.method_high_confidence_explicit,
            'subtle_toxicity': self.method_subtle_toxicity,
            'contextual_amplification': self.method_contextual_amplification,
            'threat_escalation': self.method_threat_escalation,
            'precision_targeting': self.method_precision_targeting,
            'baseline': lambda x: x  # No tagging
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")
        
        tagging_func = method_map[method]
        return [tagging_func(text) for text in tqdm(texts, desc=f"Applying {method}")]
    
    def evaluate_tagging_method(self, 
                               texts: List[str], 
                               labels: np.ndarray,
                               method: str,
                               max_length: int = 256,
                               batch_size: int = 16) -> Dict:
        """
        Evaluate an improved tagging method by comparing performance with baseline.
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING IMPROVED METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        # Apply tagging method
        tagged_texts = self.apply_tagging_method(texts, method)
        
        # Show examples of transformations
        transformations = 0
        print(f"\nExample transformations:")
        for i in range(min(10, len(texts))):
            if texts[i] != tagged_texts[i]:
                print(f"Original: {texts[i]}")
                print(f"Tagged:   {tagged_texts[i]}")
                print()
                transformations += 1
                if transformations >= 3:  # Show max 3 examples
                    break
        
        if transformations == 0:
            print("No transformations applied (method was conservative)")
        else:
            transformation_rate = (transformations / len(texts)) * 100
            print(f"Transformation rate: {transformation_rate:.1f}%")
        
        # Load model and evaluate
        model = self.load_model()
        predictions = []
        
        for i in tqdm(range(0, len(tagged_texts), batch_size), desc="Evaluating"):
            batch_texts = tagged_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate metrics
        results = self._calculate_metrics(labels, predictions, method)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, method: str) -> Dict:
        """Calculate comprehensive metrics for evaluation."""
        results = {'method': method}
        
        # Per-label AUC
        label_aucs = []
        print("\nPer-label AUC scores:")
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                label_aucs.append(auc)
                results[f'auc_{label}'] = auc
                print(f"  {label:15}: {auc:.4f}")
            else:
                label_aucs.append(0.0)
                results[f'auc_{label}'] = 0.0
                print(f"  {label:15}: No positive samples")
        
        # Mean AUC
        valid_aucs = [auc for auc in label_aucs if auc > 0]
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        results['mean_auc'] = mean_auc
        print(f"\n  Mean AUC: {mean_auc:.4f}")
        
        # Neutral performance
        neutral_metrics = self._evaluate_neutral_performance(y_true, y_pred)
        results.update(neutral_metrics)
        
        print(f"  Neutral accuracy: {neutral_metrics['neutral_accuracy']:.4f}")
        print(f"  Neutral FP rate: {neutral_metrics['neutral_fp_rate']:.4f}")
        print(f"  Neutral samples: {neutral_metrics['neutral_total']}")
        
        return results
    
    def _evaluate_neutral_performance(self, y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict:
        """Evaluate performance on neutral comments."""
        y_pred_bin = (y_pred_probs > threshold).astype(int)
        neutral_mask = (y_true.sum(axis=1) == 0)
        neutral_total = neutral_mask.sum()
        
        if neutral_total == 0:
            return {
                'neutral_total': 0,
                'neutral_correct': 0,
                'neutral_accuracy': 0.0,
                'neutral_fp_rate': 0.0
            }
        
        neutral_pred_mask = (y_pred_bin[neutral_mask].sum(axis=1) == 0)
        neutral_correct = neutral_pred_mask.sum()
        neutral_fp = neutral_total - neutral_correct
        
        return {
            'neutral_total': int(neutral_total),
            'neutral_correct': int(neutral_correct),
            'neutral_fp': int(neutral_fp),
            'neutral_accuracy': float(neutral_correct / neutral_total),
            'neutral_fp_rate': float(neutral_fp / neutral_total)
        }
    
    def compare_all_methods(self, 
                           texts: List[str], 
                           labels: np.ndarray,
                           methods: List[str] = None) -> pd.DataFrame:
        """
        Compare all improved tagging methods and return results DataFrame.
        """
        if methods is None:
            methods = [
                'baseline',
                'high_confidence_explicit',
                'subtle_toxicity', 
                'contextual_amplification',
                'threat_escalation',
                'precision_targeting'
            ]
        
        results = []
        
        with mlflow.start_run(run_name="improved_tagging_methods_comparison"):
            for method in methods:
                print(f"\n{'='*80}")
                print(f"EVALUATING IMPROVED METHOD: {method.upper()}")
                print(f"{'='*80}")
                
                try:
                    method_results = self.evaluate_tagging_method(texts, labels, method)
                    results.append(method_results)
                    
                    # Log to MLflow
                    with mlflow.start_run(run_name=f"improved_{method}", nested=True):
                        for metric, value in method_results.items():
                            if isinstance(value, (int, float)) and metric != 'method':
                                mlflow.log_metric(metric, value)
                        mlflow.log_param('tagging_method', method)
                        mlflow.log_param('approach', 'improved_conservative')
                        mlflow.log_param('version', '2.0')
                
                except Exception as e:
                    print(f"❌ Error evaluating {method}: {str(e)}")
                    continue
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mean_auc', ascending=False)
        
        # Display results
        self._display_comparison_results(results_df)
        
        return results_df
    
    def _display_comparison_results(self, results_df: pd.DataFrame):
        """Display formatted comparison results."""
        print(f"\n{'='*100}")
        print("IMPROVED TAGGING METHODS COMPARISON RESULTS")
        print(f"{'='*100}")
        
        print(f"{'Method':<25} {'Mean AUC':<12} {'Neutral Acc':<12} {'Neutral FP':<12} {'Improvement':<12}")
        print("-" * 100)
        
        baseline_auc = results_df[results_df['method'] == 'baseline']['mean_auc'].iloc[0] if 'baseline' in results_df['method'].values else 0
        
        for _, row in results_df.iterrows():
            improvement = ""
            if row['method'] != 'baseline' and baseline_auc > 0:
                imp_pct = ((row['mean_auc'] - baseline_auc) / baseline_auc) * 100
                improvement = f"{imp_pct:+.2f}%"
            
            print(f"{row['method']:<25} {row['mean_auc']:<12.4f} {row['neutral_accuracy']:<12.4f} "
                  f"{row['neutral_fp_rate']:<12.4f} {improvement:<12}")
        
        # Show best improvements per label
        print(f"\n{'='*100}")
        print("BEST IMPROVEMENTS BY TOXICITY TYPE")
        print(f"{'='*100}")
        
        if baseline_auc > 0:
            baseline_row = results_df[results_df['method'] == 'baseline'].iloc[0]
            
            for label in self.labels:
                auc_col = f'auc_{label}'
                if auc_col in results_df.columns:
                    best_row = results_df.loc[results_df[auc_col].idxmax()]
                    if best_row['method'] != 'baseline':
                        baseline_score = baseline_row[auc_col]
                        best_score = best_row[auc_col]
                        improvement = ((best_score - baseline_score) / baseline_score) * 100
                        
                        print(f"{label.upper():15} | Best: {best_row['method']:<25} | "
                              f"AUC: {best_score:.4f} | Improvement: {improvement:+.1f}%")


# ============================================================================
# USAGE DEMONSTRATION AND TESTING
# ============================================================================

def demo_improved_methods():
    """Demonstrate the improved tagging methods with examples."""
    tagger = ImprovedToxicCommentTagger()
    
    # Test cases designed to show different types of toxicity
    test_cases = [
        # High-confidence explicit cases
        ("I will kill you if you don't shut up", "Explicit threat"),
        ("You fucking piece of shit", "Severe profanity"),
        ("Die you stupid nigger", "Identity slur + threat"),
        
        # Subtle toxicity cases
        ("Oh wow, great job genius", "Sarcastic insult"),
        ("No offense but you people are all the same", "Passive-aggressive + coded"),
        ("Just saying, but urban youth are the problem", "Coded language"),
        
        # Amplification cases
        ("YOU ARE SO FUCKING STUPID!!!", "Amplified toxicity"),
        ("I HATE HATE HATE you sooo much", "Multiple amplifiers"),
        
        # Escalation cases
        ("I am so tired of you, this is your last warning", "Threat escalation"),
        ("You better stop or you'll regret it", "Warning escalation"),
        
        # Targeting cases
        ("You people like you don't belong here", "Group targeting"),
        ("Get out of here, we don't want your kind", "Exclusionary targeting"),
        
        # Neutral cases (should not be tagged)
        ("I disagree with your opinion", "Neutral disagreement"),
        ("The weather is nice today", "Completely neutral"),
        ("I don't understand this topic", "Neutral confusion")
    ]
    
    methods = [
        ('High Confidence', tagger.method_high_confidence_explicit),
        ('Subtle Toxicity', tagger.method_subtle_toxicity),
        ('Amplification', tagger.method_contextual_amplification),
        ('Escalation', tagger.method_threat_escalation),
        ('Targeting', tagger.method_precision_targeting)
    ]
    
    print("IMPROVED TAGGING METHODS DEMONSTRATION")
    print("=" * 80)
    print()
    
    for text, description in test_cases:
        print(f"Text: {text}")
        print(f"Type: {description}")
        print("-" * 60)
        
        any_tagged = False
        for method_name, method_func in methods:
            try:
                tagged = method_func(text)
                if tagged != text:  # Only show if it was tagged
                    print(f"{method_name:18}: {tagged}")
                    any_tagged = True
            except Exception as e:
                print(f"{method_name:18}: Error - {str(e)}")
        
        if not any_tagged:
            print(f"{'No tagging':18}: {text} (conservative approach)")
        
        print()

if __name__ == "__main__":
    demo_improved_methods()