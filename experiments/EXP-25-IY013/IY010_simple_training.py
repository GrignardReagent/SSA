#!/usr/bin/env python3
"""Ultra-simplified TF Transformer training script.

This version uses the new data loading utilities and built-in training methods
for maximum simplicity and reliability. Perfect for quick experiments and
as a template for your own training scripts.
"""

from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.TF_transformer import TFTransformer, ModelCfg
from utils.data_loaders import quick_load_tf_data

def main():
    """Simplified training pipeline."""
    
    print("🚀 Simple TF Transformer Training")
    print("=" * 40)
    
    # Configuration
    BASE_DIR = Path(__file__).resolve().parent
    SYNTHETIC_DIR = BASE_DIR / "data"
    EXPERIMENTAL_TSV = BASE_DIR / "exp_data" / "19316_2020_10_26_steadystate_glucose_144m_2w2_00_post_media_switch.tsv"
    
    # 1. Load synthetic data for pretraining (if available)
    if SYNTHETIC_DIR.exists() and list(SYNTHETIC_DIR.glob("*.csv")):
        print("📂 Loading synthetic data for pretraining...")
        
        train_loader, val_loader, n_classes = quick_load_tf_data(
            SYNTHETIC_DIR, 
            batch_size=32, 
            val_split=0.2,
            verbose=True
        )
        
        # Create model
        cfg = ModelCfg(
            n_classes=n_classes,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_len=1024,
            verbose=True,
            learning_rate=1e-3,
            optimizer='AdamW'
        )
        model = TFTransformer(cfg)
        
        # Pretrain
        print("\n🎭 Pretraining on synthetic data...")
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            patience=5,
            save_path=str(BASE_DIR / "pretrained_model.pt")
        )
        print("✅ Pretraining complete!")
        
    else:
        print("⚠️  No synthetic data found, creating model without pretraining")
        cfg = ModelCfg(n_classes=2, verbose=True)  # Default binary classification
        model = TFTransformer(cfg)
    
    # 2. Fine-tune on experimental data (if available)
    if EXPERIMENTAL_TSV.exists():
        print("\n📂 Loading experimental data for fine-tuning...")
        
        try:
            train_loader, val_loader, exp_classes = quick_load_tf_data(
                EXPERIMENTAL_TSV,
                batch_size=32,
                val_split=0.2,
                verbose=True
            )
            
            # Update model for experimental classes if needed
            if exp_classes != model.cfg.n_classes:
                print(f"🔄 Updating model from {model.cfg.n_classes} to {exp_classes} classes")
                # Recreate model with correct number of classes
                cfg.n_classes = exp_classes
                new_model = TFTransformer(cfg)
                # Transfer pretrained weights if available
                if (BASE_DIR / "pretrained_model.pt").exists():
                    new_model.load_pretrained_encoder(str(BASE_DIR / "pretrained_model.pt"))
                model = new_model
            
            # Setup for fine-tuning
            model.freeze_encoder(freeze=True)
            model.reset_classifier()
            model.cfg.learning_rate = 1e-4  # Lower LR for fine-tuning
            model._setup_training()
            
            print("\n🔧 Fine-tuning on experimental data...")
            model.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=15,
                patience=5,
                save_path=str(BASE_DIR / "finetuned_model.pt")
            )
            print("✅ Fine-tuning complete!")
            
        except Exception as e:
            print(f"❌ Error loading experimental data: {e}")
    
    else:
        print("⚠️  No experimental data found, skipping fine-tuning")
    
    print("\n🎉 Training pipeline complete!")
    print(f"💾 Models saved in: {BASE_DIR}")


if __name__ == "__main__":
    main()
