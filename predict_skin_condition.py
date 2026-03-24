"""
Unified Skin Condition Classification Predictor
Supports both TensorFlow and PyTorch models with ensemble capabilities
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import paths
try:
    from data_cleaning.paths import DATA, TENSORFLOW_MODEL_FOLDER, PYTORCH_MODEL_FOLDER, METRICS_FOLDER
except ImportError:
    # Fallback paths
    DATA = os.path.join(parent_dir, "data")
    TENSORFLOW_MODEL_FOLDER = os.path.join(parent_dir, "outputs/saved_models/tensorflow_model_metrics")
    PYTORCH_MODEL_FOLDER = os.path.join(parent_dir, "outputs/saved_models/pytorch_model_metrics")
    METRICS_FOLDER = os.path.join(parent_dir, "metrics")

# TensorFlow imports (optional)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available. TensorFlow models will not be loadable.")

# PyTorch imports (optional)
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    PT_AVAILABLE = True
except ImportError:
    PT_AVAILABLE = False
    print("⚠️ PyTorch not available. PyTorch models will not be loadable.")

# Constants
IMG_SIZE = 224

# Default class names (adjust based on your dataset)
train_df = pd.read_csv(os.path.join(DATA, "train/split_train.csv"))
DEFAULT_CLASS_NAMES = sorted(train_df["label_name"].unique())

class EfficientNetB0PyTorch(nn.Module):
    """PyTorch model architecture matching training"""
    def __init__(self, num_classes=8):
        super().__init__()
        from torchvision import models
        self.base_model = models.efficientnet_b0(pretrained=False)
        
        # Modify classifier head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

class SkinConditionPredictor:
    """
    Unified predictor for skin condition classification
    Supports both TensorFlow and PyTorch models with ensemble
    """
    
    def __init__(self, 
                 model_type: str = 'auto',
                 tf_model_path: Optional[str] = None,
                 pt_model_path: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 threshold: float = 0.5,
                 device: str = 'auto',
                 ensemble_method: str = 'weighted'):
        """
        Initialize the predictor
        
        Args:
            model_type: 'tensorflow', 'pytorch', 'ensemble', or 'auto'
            tf_model_path: Path to TensorFlow model (.h5 or SavedModel)
            pt_model_path: Path to PyTorch model weights (.pth file)
            class_names: List of class names (if None, will try to load from metrics)
            threshold: Confidence threshold for predictions
            device: 'cuda', 'cpu', or 'auto' (for PyTorch)
            ensemble_method: 'average', 'max', or 'weighted' (for ensemble mode)
        """
        self.threshold = threshold
        self.ensemble_method = ensemble_method
        
        # Load class names
        if class_names is None:
            self.class_names = self._load_class_names()
        else:
            self.class_names = class_names
        
        # Ensure class_names is a list, not a set
        if isinstance(self.class_names, set):
            self.class_names = sorted(list(self.class_names))
        
        self.num_classes = len(self.class_names)
        
        # Set device for PyTorch
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if PT_AVAILABLE else 'cpu'
        else:
            self.device = torch.device(device)
        
        # Load models
        self.tf_model = None
        self.pt_model = None
        self.model_type = model_type
        
        if model_type == 'auto':
            self._load_auto_model(tf_model_path, pt_model_path)
        elif model_type == 'tensorflow':
            self._load_tensorflow_model(tf_model_path)
        elif model_type == 'pytorch':
            self._load_pytorch_model(pt_model_path)
        elif model_type == 'ensemble':
            # Try to load both models
            tf_loaded = False
            pt_loaded = False
            
            try:
                self._load_tensorflow_model(tf_model_path)
                tf_loaded = True
            except Exception as e:
                print(f"⚠️ Could not load TensorFlow model: {e}")
            
            try:
                self._load_pytorch_model(pt_model_path)
                pt_loaded = True
            except Exception as e:
                print(f"⚠️ Could not load PyTorch model: {e}")
            
            if not tf_loaded and not pt_loaded:
                raise RuntimeError("Could not load any model for ensemble")
            
            self.model_type = 'ensemble'
            print(f"✅ Ensemble mode ready (TF: {tf_loaded}, PT: {pt_loaded})")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Load preprocessing transforms for PyTorch
        if self.pt_model is not None:
            self.pt_transform = self._create_pt_transform()
        
        print(f"\n✅ Predictor initialized successfully")
        print(f"   Model Type: {self.model_type.upper()}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Threshold: {self.threshold}")
        if self.model_type == 'ensemble':
            print(f"   Ensemble Method: {self.ensemble_method}")
    
    def _load_class_names(self) -> List[str]:
        """Load class names from saved metrics or training data"""
        try:
            # Try to load from metrics file
            metrics_path = os.path.join(METRICS_FOLDER, "tensorflow_model_metrics.joblib")
            if os.path.exists(metrics_path):
                metrics = joblib.load(metrics_path)
                if 'class_names' in metrics:
                    class_names = metrics['class_names']
                    if isinstance(class_names, set):
                        class_names = sorted(list(class_names))
                    return class_names
            
            # Try PyTorch metrics
            pt_metrics_path = os.path.join(PYTORCH_MODEL_FOLDER, "pytorch_model_metrics.joblib")
            if os.path.exists(pt_metrics_path):
                metrics = joblib.load(pt_metrics_path)
                if 'class_names' in metrics:
                    class_names = metrics['class_names']
                    if isinstance(class_names, set):
                        class_names = sorted(list(class_names))
                    return class_names
            
            # Try to load from training data
            train_path = os.path.join(DATA, "train/split_train.csv")
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
                if 'label_name' in train_df.columns:
                    class_names = sorted(train_df['label_name'].unique())
                    if isinstance(class_names, set):
                        class_names = sorted(list(class_names))
                    return class_names
            
            # Fallback to default
            print(f"⚠️ Using default class names: {DEFAULT_CLASS_NAMES}")
            return DEFAULT_CLASS_NAMES
            
        except Exception as e:
            print(f"⚠️ Could not load class names: {e}")
            return DEFAULT_CLASS_NAMES
    
    def _load_auto_model(self, tf_path, pt_path):
        """Auto-detect and load available model - try both TensorFlow and PyTorch"""
        loaded = False
        
        # Try TensorFlow first
        if TF_AVAILABLE:
            try:
                self._load_tensorflow_model(tf_path)
                self.model_type = 'tensorflow'
                loaded = True
                print("✅ Using TensorFlow model")
            except Exception as e:
                print(f"⚠️ Could not load TensorFlow model: {e}")
        
        # Try PyTorch if TensorFlow failed
        if not loaded and PT_AVAILABLE:
            try:
                self._load_pytorch_model(pt_path)
                self.model_type = 'pytorch'
                loaded = True
                print("✅ Using PyTorch model")
            except Exception as e:
                print(f"⚠️ Could not load PyTorch model: {e}")
        
        # If both loaded successfully, switch to ensemble
        if self.tf_model is not None and self.pt_model is not None:
            self.model_type = 'ensemble'
            print("✅ Both models loaded - Ensemble mode available")
        
        if not loaded:
            raise RuntimeError("Could not load any model. Please check model paths.")
    
    def _load_tensorflow_model(self, model_path: Optional[str] = None):
        """Load TensorFlow model - supports both .h5 and SavedModel formats"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        
        if model_path is None:
            # Look for model files in the TensorFlow folder
            if os.path.exists(TENSORFLOW_MODEL_FOLDER):
                # First look for .h5 files (Keras format)
                h5_files = [f for f in os.listdir(TENSORFLOW_MODEL_FOLDER) if f.endswith('.h5')]
                if h5_files:
                    # Sort by modification time to get latest
                    h5_files.sort(key=lambda f: os.path.getmtime(os.path.join(TENSORFLOW_MODEL_FOLDER, f)))
                    model_path = os.path.join(TENSORFLOW_MODEL_FOLDER, h5_files[-1])
                    print(f"📦 Found TensorFlow .h5 model: {model_path}")
                else:
                    # Look for SavedModel directories
                    model_dirs = [d for d in os.listdir(TENSORFLOW_MODEL_FOLDER) 
                                 if os.path.isdir(os.path.join(TENSORFLOW_MODEL_FOLDER, d))]
                    if model_dirs:
                        model_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(TENSORFLOW_MODEL_FOLDER, d)))
                        model_path = os.path.join(TENSORFLOW_MODEL_FOLDER, model_dirs[-1])
                        print(f"📦 Found TensorFlow SavedModel: {model_path}")
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"TensorFlow model not found at {model_path}")
        
        print(f"📦 Loading TensorFlow model from: {model_path}")
        
        try:
            # Try loading as Keras .h5 model first
            if model_path.endswith('.h5'):
                self.tf_model = tf.keras.models.load_model(model_path)
                print("✅ TensorFlow Keras model loaded successfully")
            else:
                # Try loading as SavedModel
                self.tf_model = tf.saved_model.load(model_path)
                print("✅ TensorFlow SavedModel loaded successfully")
            
            # Verify model works
            if hasattr(self.tf_model, 'predict'):
                print("   Model is ready for predictions")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {e}")
    
    def _load_pytorch_model(self, model_path: Optional[str] = None):
        """Load PyTorch model"""
        if not PT_AVAILABLE:
            raise ImportError("PyTorch is not installed")
        
        if model_path is None:
            # Try to find the best model
            best_model = os.path.join(PYTORCH_MODEL_FOLDER, 'best_model_gradual.pth')
            if os.path.exists(best_model):
                model_path = best_model
            else:
                # Look for any .pth file
                pth_files = [f for f in os.listdir(PYTORCH_MODEL_FOLDER) if f.endswith('.pth')]
                if not pth_files:
                    raise FileNotFoundError(f"No PyTorch models found in {PYTORCH_MODEL_FOLDER}")
                model_path = os.path.join(PYTORCH_MODEL_FOLDER, pth_files[-1])
        
        print(f"📦 Loading PyTorch model from: {model_path}")
        
        # Initialize model
        self.pt_model = EfficientNetB0PyTorch(num_classes=self.num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.pt_model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_acc' in checkpoint:
                print(f"   Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            self.pt_model.load_state_dict(checkpoint)
        
        self.pt_model = self.pt_model.to(self.device)
        self.pt_model.eval()
        print(f"✅ PyTorch model loaded on {self.device}")
    
    def _create_pt_transform(self):
        """Create PyTorch preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_tf_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for TensorFlow model"""
        # Resize
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32)
        
        # EfficientNet preprocessing
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _preprocess_pt_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for PyTorch model"""
        # Apply transforms
        img_tensor = self.pt_transform(image)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict_tensorflow(self, image: Image.Image) -> Tuple[np.ndarray, Dict]:
        """Make prediction using TensorFlow model"""
        # Preprocess
        img_array = self._preprocess_tf_image(image)
        
        # Get prediction
        predictions = self.tf_model.predict(img_array, verbose=0)
        
        # Get probabilities
        if len(predictions.shape) > 1:
            probs = predictions[0]
        else:
            probs = predictions
        
        # Apply softmax if needed
        if not np.isclose(np.sum(probs), 1.0):
            probs = tf.nn.softmax(probs).numpy()
        
        return probs, {'model': 'tensorflow'}
    
    def predict_pytorch(self, image: Image.Image) -> Tuple[np.ndarray, Dict]:
        """Make prediction using PyTorch model"""
        # Preprocess
        img_tensor = self._preprocess_pt_image(image)
        
        # Get prediction
        with torch.no_grad():
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext():
                outputs = self.pt_model(img_tensor)
                probs = torch.softmax(outputs[0], dim=0).cpu().numpy()
        
        return probs, {'model': 'pytorch'}
    
    def predict_ensemble(self, image: Image.Image, method: str = None) -> Tuple[np.ndarray, Dict]:
        """Make ensemble prediction using both models"""
        if method is None:
            method = self.ensemble_method
        
        # Get predictions from both models
        tf_probs = None
        pt_probs = None
        models_used = []
        
        if self.tf_model is not None:
            tf_probs, _ = self.predict_tensorflow(image)
            models_used.append('tensorflow')
        
        if self.pt_model is not None:
            pt_probs, _ = self.predict_pytorch(image)
            models_used.append('pytorch')
        
        # If only one model is available, return its prediction
        if tf_probs is None and pt_probs is None:
            raise RuntimeError("No models available for ensemble")
        elif tf_probs is None:
            return pt_probs, {'model': 'pytorch_only', 'ensemble_method': method}
        elif pt_probs is None:
            return tf_probs, {'model': 'tensorflow_only', 'ensemble_method': method}
        
        # Combine predictions
        if method == 'average':
            combined_probs = (tf_probs + pt_probs) / 2
        elif method == 'max':
            combined_probs = np.maximum(tf_probs, pt_probs)
        elif method == 'weighted':
            # Weight based on model confidence
            tf_confidence = np.max(tf_probs)
            pt_confidence = np.max(pt_probs)
            total = tf_confidence + pt_confidence
            if total > 0:
                combined_probs = (tf_probs * tf_confidence + pt_probs * pt_confidence) / total
            else:
                combined_probs = (tf_probs + pt_probs) / 2
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return combined_probs, {
            'model': 'ensemble',
            'ensemble_method': method,
            'models_used': models_used,
            'tensorflow_probs': tf_probs,
            'pytorch_probs': pt_probs
        }
    
    def predict(self, 
                image: Union[str, Image.Image, np.ndarray],
                return_details: bool = False,
                ensemble_method: str = None) -> Union[str, Dict]:
        """
        Make prediction for a single image
        
        Args:
            image: Image path, PIL Image, or numpy array
            return_details: If True, return detailed prediction info
            ensemble_method: Override ensemble method for this prediction
            
        Returns:
            Predicted class name or detailed dictionary
        """
        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Make prediction based on model type
        if self.model_type == 'ensemble':
            probs, meta = self.predict_ensemble(image, method=ensemble_method)
        elif self.model_type == 'tensorflow':
            if self.tf_model is None:
                raise RuntimeError("TensorFlow model not loaded")
            probs, meta = self.predict_tensorflow(image)
        elif self.model_type == 'pytorch':
            if self.pt_model is None:
                raise RuntimeError("PyTorch model not loaded")
            probs, meta = self.predict_pytorch(image)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Ensure probs is 1D array
        if len(probs.shape) > 1:
            probs = probs.flatten()
        
        # Get top prediction
        predicted_idx = np.argmax(probs)
        predicted_class = self.class_names[predicted_idx] if predicted_idx < len(self.class_names) else "Unknown"
        confidence = probs[predicted_idx]
        
        # Get top-k predictions
        top_k = min(5, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            {
                'class': self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                'probability': float(probs[i]),
                'confidence_percentage': f"{probs[i]*100:.2f}%"
            }
            for i in top_indices
        ]
        
        if return_details:
            result = {
                'predicted_class': predicted_class,
                'predicted_index': int(predicted_idx),
                'confidence': float(confidence),
                'is_confident': confidence >= self.threshold,
                'all_probabilities': {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(probs[i]) 
                    for i in range(len(probs))
                },
                'top_predictions': top_predictions,
                'threshold_used': self.threshold,
                'model_used': meta.get('model', self.model_type)
            }
            
            # Add ensemble details if applicable
            if self.model_type == 'ensemble' and 'tensorflow_probs' in meta:
                result['ensemble_method'] = meta.get('ensemble_method', self.ensemble_method)
                result['models_used'] = meta.get('models_used', [])
                result['tensorflow_probs'] = {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(meta['tensorflow_probs'][i])
                    for i in range(len(meta['tensorflow_probs']))
                }
                result['pytorch_probs'] = {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(meta['pytorch_probs'][i])
                    for i in range(len(meta['pytorch_probs']))
                }
            
            return result
        
        return predicted_class
    
    def predict_batch(self, 
                      images: List[Union[str, Image.Image]],
                      return_details: bool = False) -> List[Union[str, Dict]]:
        """
        Predict multiple images
        
        Args:
            images: List of image paths or PIL Images
            return_details: If True, return detailed predictions
            
        Returns:
            List of predictions
        """
        results = []
        for image in images:
            result = self.predict(image, return_details=return_details)
            results.append(result)
        return results
    
    def predict_with_visualization(self, 
                                   image: Union[str, Image.Image],
                                   save_path: Optional[str] = None) -> Dict:
        """
        Predict and visualize the result
        
        Args:
            image: Image path or PIL Image
            save_path: If provided, save visualization to this path
            
        Returns:
            Prediction details
        """
        # Get prediction
        result = self.predict(image, return_details=True)
        
        # Load image
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title(f"Input Image\nPrediction: {result['predicted_class']}\n"
                         f"Confidence: {result['confidence']:.2%}")
        
        # Create bar chart of top predictions
        top_preds = result['top_predictions'][:5]
        classes = [p['class'] for p in top_preds]
        probs = [p['probability'] for p in top_preds]
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(probs))]
        
        axes[1].barh(classes, probs, color=colors)
        axes[1].set_xlabel('Probability')
        axes[1].set_title(f'Top Predictions (Model: {result["model_used"]})')
        axes[1].axvline(x=self.threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.threshold})')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()
        
        return result


class nullcontext:
    """Context manager that does nothing"""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# Convenience functions
def load_predictor(model_type='auto', threshold=0.5, ensemble_method='weighted'):
    """Quick load predictor"""
    return SkinConditionPredictor(
        model_type=model_type, 
        threshold=threshold,
        ensemble_method=ensemble_method
    )


def predict_image(image_path, predictor=None, threshold=0.5):
    """Quick prediction"""
    if predictor is None:
        predictor = load_predictor(threshold=threshold)
    return predictor.predict(image_path)


def predict_with_ensemble(image_path, method='weighted', threshold=0.5):
    """Quick ensemble prediction"""
    predictor = load_predictor(model_type='ensemble', threshold=threshold, ensemble_method=method)
    return predictor.predict(image_path, return_details=True)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("SKIN CONDITION PREDICTOR - TEST")
    print("=" * 60)
    
    # Test with a sample image (replace with actual path)
    test_image = "data/train/AK/ISIC_0026848.jpg"
    
    # Check if test image exists
    if os.path.exists(test_image):
        try:
            # Load predictor with ensemble mode
            predictor = load_predictor(model_type='auto', threshold=0.5, ensemble_method='weighted')
            
            # Single prediction
            print("\n🔮 Ensemble Prediction:")
            result = predictor.predict(test_image, return_details=True)
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Confident: {result['is_confident']}")
            print(f"   Model Used: {result['model_used']}")
            
            if result['model_used'] == 'ensemble':
                print(f"   Ensemble Method: {result.get('ensemble_method', 'N/A')}")
                print(f"   Models Used: {result.get('models_used', [])}")
            
            # Show top predictions
            print(f"\n📈 Top Predictions:")
            for pred in result['top_predictions'][:3]:
                print(f"   {pred['class']}: {pred['confidence_percentage']}")
            
            # Show individual model predictions if available
            if 'tensorflow_probs' in result:
                print(f"\n🤖 Individual Model Predictions:")
                tf_best = max(result['tensorflow_probs'].items(), key=lambda x: x[1])
                pt_best = max(result['pytorch_probs'].items(), key=lambda x: x[1])
                print(f"   TensorFlow: {tf_best[0]} ({tf_best[1]:.2%})")
                print(f"   PyTorch: {pt_best[0]} ({pt_best[1]:.2%})")
            
            # Visualize
            predictor.predict_with_visualization(test_image)
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️ Test image not found: {test_image}")
        print("Please update the path to your test image")
        print("\nYou can test with:")
        print("  predictor = load_predictor()")
        print("  result = predictor.predict('your_image.jpg', return_details=True)")