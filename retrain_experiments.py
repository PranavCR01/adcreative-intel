"""
Retrain CLIP + Weibull model with cleaned data.

This script will:
1. Load one of the cleaned datasets from audit_and_fix.py
2. Train using your existing setup (CLIP-ViT-B/32 frozen, multi-task BCE + Weibull NLL)
3. Evaluate on test set and report Spearman r
4. Save model and metrics

Usage:
  python retrain_experiments.py --dataset exp1  # No sentinels
  python retrain_experiments.py --dataset exp2  # Balanced 60/40
  python retrain_experiments.py --dataset exp3  # Censored (advanced)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from transformers import CLIPProcessor
from tqdm import tqdm
import json
import os
from datetime import datetime
from model.clip_head import CreativeScorer
from model.survival import WeibullNLLLoss

# ============================================
# CONFIG
# ============================================

class Config:
    # CLIP processor (model arch is fixed inside CreativeScorer)
    clip_model = "openai/clip-vit-base-patch32"

    # Training
    lr = 5e-4
    batch_size = 64
    epochs = 30
    weight_decay = 1e-4

    # Data
    image_col = 'image_path'  # Update if different

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __repr__(self):
        return json.dumps(vars(self), indent=2)

# ============================================
# DATASET
# ============================================

class AdDataset(Dataset):
    def __init__(self, df, processor, config):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.config = config

        # Identify columns
        self.ctr_col = self._find_col(['ctr_score', 'ctr'])
        self.halflife_col = self._find_col(['halflife', 'half_life'])
        self.image_col = config.image_col
        self.censored_col = 'is_censored' if 'is_censored' in df.columns else None

        # Filter out rows with missing CTR (for exp3 censored data)
        if 'is_censored' in df.columns:
            self.df = df[df['is_censored'] == 0].reset_index(drop=True)
            print(f"Filtered {len(df) - len(self.df)} censored rows from training")

    def _find_col(self, candidates):
        for col in candidates:
            if col in self.df.columns:
                return col
        for col in self.df.columns:
            for cand in candidates:
                if cand.lower() in col.lower():
                    return col
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        from PIL import Image
        try:
            image = Image.open(row[self.image_col]).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='gray')

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        ctr_score = float(row[self.ctr_col]) if self.ctr_col else 0.5
        halflife = float(row[self.halflife_col]) if self.halflife_col and pd.notna(row[self.halflife_col]) else 1.0
        censored = bool(row[self.censored_col]) if self.censored_col else False

        return {
            'pixel_values': pixel_values,
            'ctr_score': torch.tensor(ctr_score, dtype=torch.float32),
            'halflife': torch.tensor(halflife, dtype=torch.float32),
            'censored': torch.tensor(censored, dtype=torch.bool),
        }

# ============================================
# LOSS
# ============================================

_weibull_loss_fn = WeibullNLLLoss()

def combined_loss(outputs, ctr_score, halflife, censored):
    # Loss = 0.5 * BCELoss(ctr) + 0.5 * WeibullNLLLoss(fatigue)
    bce_loss = nn.functional.binary_cross_entropy(
        outputs['ctr_score'].squeeze(-1), ctr_score
    )
    weibull_loss = _weibull_loss_fn(outputs['weibull_params'], halflife, censored)
    total_loss = 0.5 * bce_loss + 0.5 * weibull_loss
    return total_loss, {'bce': bce_loss.item(), 'weibull': weibull_loss.item()}

# ============================================
# TRAINING
# ============================================

def train_epoch(model, loader, optimizer, config):
    model.train()
    total_loss = 0
    metrics = {'bce': 0, 'weibull': 0}

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(config.device)
        ctr_score = batch['ctr_score'].to(config.device)
        halflife = batch['halflife'].to(config.device)
        censored = batch['censored'].to(config.device)

        optimizer.zero_grad()

        outputs = model(pixel_values)
        loss, loss_dict = combined_loss(outputs, ctr_score, halflife, censored)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            metrics[k] += v

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(loader)
    return total_loss / n, {k: v/n for k, v in metrics.items()}

def evaluate(model, loader, config):
    model.eval()

    all_ctr_pred = []
    all_ctr_true = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            pixel_values = batch['pixel_values'].to(config.device)
            ctr_score = batch['ctr_score']

            outputs = model(pixel_values)

            all_ctr_pred.extend(outputs['ctr_score'].squeeze(-1).cpu().numpy())
            all_ctr_true.extend(ctr_score.numpy())

    spearman_r, spearman_p = spearmanr(all_ctr_true, all_ctr_pred)
    mse = np.mean((np.array(all_ctr_true) - np.array(all_ctr_pred)) ** 2)

    return {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mse': mse,
    }

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['exp1', 'exp2', 'exp3'],
                       help='Which experiment dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Override data path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    print("=== CONFIGURATION ===")
    print(config)
    print(f"\nDevice: {config.device}")

    # Load data
    if args.data_path:
        data_path = args.data_path
    else:
        import glob
        candidates = glob.glob(f'data/*{args.dataset}*.csv')
        if not candidates:
            print(f"ERROR: Could not find {args.dataset} dataset. Run audit_and_fix.py first.")
            return
        data_path = candidates[0]

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total rows: {len(df)}")

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Load CLIP processor
    print("\nLoading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(config.clip_model)

    # Create datasets
    train_dataset = AdDataset(train_df, processor, config)
    test_dataset = AdDataset(test_df, processor, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    # Create model — uses canonical CreativeScorer (frozen CLIP + multi-task head)
    print("\nInitializing model...")
    model = CreativeScorer().to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Training loop
    print("\n=== TRAINING ===")
    best_spearman = -1
    results = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config)
        print(f"Train loss: {train_loss:.4f} | BCE: {train_metrics['bce']:.4f} | "
              f"Weibull: {train_metrics['weibull']:.4f}")

        test_metrics = evaluate(model, test_loader, config)
        print(f"Test Spearman r: {test_metrics['spearman_r']:.4f} | MSE: {test_metrics['mse']:.4f}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_spearman_r': test_metrics['spearman_r'],
            'test_mse': test_metrics['mse'],
        })

        if test_metrics['spearman_r'] > best_spearman:
            best_spearman = test_metrics['spearman_r']
            model_path = f"model_{args.dataset}_best.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model: {model_path}")

    # Final evaluation
    print("\n=== FINAL RESULTS ===")
    model.load_state_dict(torch.load(f"model_{args.dataset}_best.pt"))
    final_metrics = evaluate(model, test_loader, config)

    print(f"\nBest Test Spearman r: {final_metrics['spearman_r']:.4f} (p={final_metrics['spearman_p']:.4e})")
    print(f"Test MSE: {final_metrics['mse']:.4f}")

    # Save results
    results_path = f"results_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': vars(config),
            'dataset': args.dataset,
            'final_metrics': final_metrics,
            'training_history': results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return final_metrics['spearman_r']

if __name__ == '__main__':
    main()
