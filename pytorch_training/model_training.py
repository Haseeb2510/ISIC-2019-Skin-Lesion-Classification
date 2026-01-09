import os
import sys
import time
# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
from data_cleaning.paths import DATA, PYTORCH_MODEL_FOLDER, METRICS_FOLDER, CONFUSION_METRICS
import warnings

# ========== OPTIMIZATION SETTINGS ==========
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
    torch.set_float32_matmul_precision('high')

class SkinConditionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Pre-filter valid files
        self.valid_indices = []
        
        # Verify files exist during initialization
        print("Verifying image files...")
        for idx in tqdm(range(len(self.df)), desc="Checking files"):
            image_path = self.df.iloc[idx]['filepath']
            if os.path.exists(image_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: File not found - {image_path}")
        
        print(f"Found {len(self.valid_indices)}/{len(self.df)} valid images")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Use valid indices only
        actual_idx = self.valid_indices[idx]
        image_path = self.df.iloc[actual_idx]['filepath']
        label = int(self.df.iloc[actual_idx]['label_idx'])
        
        # Load image with error handling
        try:
            # Faster image loading without verify()
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                # Pre-resize for speed
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            # Return dummy tensor (faster than PIL Image)
            dummy = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy, 0

class EfficientNetB0PyTorch(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        
        # Initially freeze ALL layers (we'll unfreeze gradually)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
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
        
        # Always keep classifier trainable
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.base_model(x)

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class SkinConditionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        
        # Gradual unfreezing configuration
        self.unfreezing_schedule = [
            (0, 7, 1.0),    # freeze most blocks
            (20, 5, 0.5),
            (40, 3, 0.2),
            (60, 1, 0.1),
            (80, 0, 0.05),
        ]
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def _print_trainable_info(self, model, stage=""):
        """Print information about trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{stage} Model Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {trainable_params/total_params*100:.1f}%")
    
    def _freeze_layers(self, model, freeze_blocks):
        blocks = list(model.base_model.features.children())

        for i, block in enumerate(blocks):
            requires_grad = (i >= freeze_blocks)
            for param in block.parameters():
                param.requires_grad = requires_grad
    
    def _adjust_learning_rates(self, model, optimizer, lr_multiplier, freeze_count):
        """Adjust learning rates while preserving optimizer state"""
        base_lr = self.config['lr']
        
        # Get the actual model (handle DataParallel)
        if hasattr(model, 'module'):
            model = model.module
        
        # Store old state before clearing
        old_state = {}
        for param in model.parameters():
            if param in optimizer.state:
                old_state[id(param)] = optimizer.state[param].copy()
        
        # Clear optimizer groups
        optimizer.param_groups.clear()
        
        # Group 1: Classifier
        classifier_params = list(model.base_model.classifier.parameters())
        optimizer.add_param_group({
            'params': classifier_params,
            'lr': base_lr * lr_multiplier,
            'name': 'classifier'
        })
        
        # Group 2: Unfrozen features
        if freeze_count < 180:
            unfrozen_features = []
            for i, param in enumerate(model.base_model.features.parameters()):
                if i >= freeze_count:
                    unfrozen_features.append(param)
            
            if unfrozen_features:
                optimizer.add_param_group({
                    'params': unfrozen_features,
                    'lr': base_lr * lr_multiplier * 0.3,
                    'name': 'unfrozen_features'
                })
        
        # Restore state for parameters that had it
        for group in optimizer.param_groups:
            for param in group['params']:
                param_id = id(param)
                if param_id in old_state:
                    optimizer.state[param] = old_state[param_id]
        
        print(f"  Created {len(optimizer.param_groups)} optimizer groups")
        for i, group in enumerate(optimizer.param_groups):
            num_params = sum(p.numel() for p in group['params'])
            print(f"    Group {i} ({group.get('name', 'unknown')}): "
                  f"LR={group['lr']:.2e}, Params={num_params:,}")

    def _apply_unfreezing_schedule_simple(self, model, epoch, optimizer):
        """Simplified version: only adjust freezing, not optimizer groups"""
        current_freeze = 180  # Default
        lr_multiplier = 1.0   # Default
        
        # Find the current schedule
        for start_epoch, freeze_layers, lr_mult in self.unfreezing_schedule:
            if epoch >= start_epoch:
                current_freeze = freeze_layers
                lr_multiplier = lr_mult
        
        # Apply freezing
        self._freeze_layers(model, current_freeze)
        
        # Adjust learning rates for existing optimizer groups
        for i, group in enumerate(optimizer.param_groups):
            if group.get('name') == 'classifier':
                group['lr'] = self.config['lr'] * lr_multiplier
            elif group.get('name') == 'unfrozen_features':
                group['lr'] = self.config['lr'] * lr_multiplier * 0.3
        
        return current_freeze, lr_multiplier

    def _print_unfreezing_info(self, epoch, freeze_count, lr_multiplier, description, model, optimizer):
        """Print unfreezing information"""
        total_layers = sum(1 for _ in model.base_model.features.parameters())
        unfrozen_layers = total_layers - freeze_count
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}: {description}")
        print(f"{'='*50}")
        print(f"  Frozen layers: {freeze_count}/{total_layers}")
        print(f"  Unfrozen layers: {unfrozen_layers}/{total_layers}")
        print(f"  Learning rate multiplier: {lr_multiplier}")
        
        # Print optimizer info
        for i, group in enumerate(optimizer.param_groups):
            lr = group['lr']
            num_params = sum(p.numel() for p in group['params'])
            print(f"  Group {i} ({group.get('name', 'unknown')}): LR={lr:.2e}, Params={num_params:,}")
        
        self._print_trainable_info(model, f"Epoch {epoch}")
    
    def create_transforms(self):
        # Training transforms (with augmentation)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=3),
            transforms.RandomResizedCrop(224, scale=(0.92, 1.0)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, 
                                 saturation=0.05, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(std=0.02)
        ])
        
        # Validation/Test transforms
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self, train_df, val_df, test_df):
        train_transform, val_transform = self.create_transforms()
        
        train_dataset = SkinConditionDataset(train_df, transform=train_transform)
        val_dataset = SkinConditionDataset(val_df, transform=val_transform)
        test_dataset = SkinConditionDataset(test_df, transform=val_transform)
        
        num_workers = min(6, os.cpu_count()) # type: ignore
        persistent_workers = True
        prefetch_factor = 3

        print(f"\nDataLoader Settings:")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Workers: {num_workers}")
        print(f"  Prefetch factor: {prefetch_factor}")
        print(f"  Persistent workers: {persistent_workers}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True, 
            pin_memory_device=str(self.device) if self.device.type == 'cuda' else ''
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )
        
        return train_loader, val_loader, test_loader
    
    def compute_class_weights(self, df):
        from sklearn.utils.class_weight import compute_class_weight
        y = df["label_idx"].values
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y),
            y=y
        )
        class_weights = np.sqrt(class_weights)
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Mixed precision backward
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar every 10 batches for speed
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': running_loss/(batch_idx+1),
                    'Acc': 100.*correct/total,
                    'Img/s': (total / (time.time() - start_time))
                })
        
        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100. * correct / max(total, 1)
        return train_loss, train_acc
    
    @torch.no_grad()
    def validate(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'Acc': 100.*correct/total})
        
        val_loss = running_loss / max(len(val_loader), 1)
        val_acc = 100. * correct / max(total, 1)
        return val_loss, val_acc
    
    def train(self, train_df, val_df, test_df):
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_df, val_df, test_df
        )
        
        # Initialize model
        num_classes = len(train_df['label_idx'].unique())
        model = EfficientNetB0PyTorch(num_classes=num_classes)
        
        # Use DataParallel if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs")
        
        model = model.to(self.device)
        
        def freeze_batchnorm(model):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

        # Print initial model info
        self._print_trainable_info(model, "Initial")
        
        # Create optimizer (initially only classifier has trainable params)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Loss function with class weights
        class_weights = self.compute_class_weights(train_df)
        criterion = FocalLoss(class_weights, gamma=2.0)
        
        # Initialize scheduler ONCE at the beginning
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True # type: ignore
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'frozen_layers': [], 'lr_multiplier': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("GRADUAL UNFREEZING TRAINING")
        print(f"{'='*60}")
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Unfreezing schedule: {self.unfreezing_schedule}")
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Apply unfreezing schedule - FIXED VERSION
            # Only modify freezing state, not optimizer groups every epoch
            current_freeze, lr_multiplier = self._apply_unfreezing_schedule_simple(model, epoch, optimizer)
            
            freeze_batchnorm(model)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Frozen layers: {current_freeze}/211, LR multiplier: {lr_multiplier}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['frozen_layers'].append(current_freeze)
            history['lr_multiplier'].append(lr_multiplier)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # GPU memory info
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB")
            
            # Early stopping
            if val_acc > best_val_acc + self.config['min_delta']:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'frozen_layers': current_freeze,
                    'config': self.config
                }, os.path.join(PYTORCH_MODEL_FOLDER, 'best_model_gradual.pth'))
                print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  ↳ No improvement for {patience_counter} epoch(s)")
                if patience_counter >= self.config['patience']:
                    print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

        
        # Load best model
        best_model_path = os.path.join(PYTORCH_MODEL_FOLDER, 'best_model_gradual.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
            print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
            print(f"Frozen layers at best: {checkpoint['frozen_layers']}")
        
        return model, history
    
    @torch.no_grad()
    def evaluate(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(self.device, non_blocking=True)
            
            with autocast():
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        print(f"Evaluation time: {time.time() - start_time:.1f}s")
        return all_labels, all_preds
    
    def plot_results(self, history, y_true, y_pred, class_names):
        # Plot training history
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        ax1.plot(history['train_acc'], label='Train Accuracy')
        ax1.plot(history['val_acc'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history['train_loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Frozen layers over time
        ax3.plot(history['frozen_layers'])
        ax3.set_title('Frozen Layers Over Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Frozen Layers')
        ax3.grid(True)
        ax3.invert_yaxis()  # So 0 is at top (all unfrozen)
        
        # LR multiplier
        ax4.plot(history['lr_multiplier'])
        ax4.set_title('Learning Rate Multiplier')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR Multiplier')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_FOLDER, 'training_gradual_unfreezing.png'))
        plt.show()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (PyTorch - Gradual Unfreezing)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFUSION_METRICS, 'confusion_matrix_gradual.png'))
        plt.show()
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT (PyTorch - Gradual Unfreezing)")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(DATA, "train/split_train.csv"))
    train_df["filepath"] = train_df.apply(
        lambda row: os.path.join(DATA, "train", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )

    val_df = pd.read_csv(os.path.join(DATA, "val/split_val.csv"))
    val_df["filepath"] = val_df.apply(
        lambda row: os.path.join(DATA, "val", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )

    test_df = pd.read_csv(os.path.join(DATA, "test/split_test.csv"))
    test_df["filepath"] = test_df.apply(
        lambda row: os.path.join(DATA, "test", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )
    
    # Verify files exist
    print("Checking file existence...")
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        existing = df["filepath"].apply(os.path.exists).sum()
        print(f"{split_name}: {existing}/{len(df)} files found")
    
    return train_df, val_df, test_df


def main():
    # Updated config for gradual unfreezing
    config = {
        'batch_size': 96,  # Reduced for more trainable parameters
        'lr': 5e-5,
        'epochs': 100,
        'patience': 15,  # Patience for gradual unfreezing
        'min_delta': 0.002,
        'weight_decay': 1e-5,
    }
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}") # type: ignore
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    train_df, val_df, test_df = load_splits()
    
    # Initialize trainer
    trainer = SkinConditionTrainer(config)
    
    # Train model
    model, history = trainer.train(train_df, val_df, test_df)
    
    # Evaluate on test set
    _, _, test_loader = trainer.create_dataloaders(train_df, val_df, test_df)
    y_true, y_pred = trainer.evaluate(model, test_loader)
    
    # Plot results
    class_names = sorted(train_df['label_name'].unique())
    trainer.plot_results(history, y_true, y_pred, class_names)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(PYTORCH_MODEL_FOLDER, 'pytorch_model_gradual.pth'))
    joblib.dump(history, os.path.join(PYTORCH_MODEL_FOLDER, 'pytorch_history_gradual.joblib'))
    
    print("\n" + "="*60)
    print("PYTORCH TRAINING WITH GRADUAL UNFREEZING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()