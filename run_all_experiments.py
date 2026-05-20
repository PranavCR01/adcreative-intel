"""
Run all 3 data cleaning experiments and compare results.

This is the master script - it will:
1. Run audit_and_fix.py to generate 3 cleaned datasets
2. Train models on each dataset
3. Compare Spearman r improvements
4. Generate final recommendation

Usage:
  python run_all_experiments.py
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
import os

def run_command(cmd, description):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f">>> Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    
    return True

def main():
    print("="*80)
    print("DATA QUALITY EXPERIMENT: Sentinel Row Removal & Retraining")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Audit and create cleaned datasets
    print("\n" + "="*80)
    print("STEP 1: Data Audit & Cleaning")
    print("="*80)
    
    if not run_command("py -3.11 audit_and_fix.py", "Running data audit and generating cleaned datasets"):
        print("\nERROR: Audit step failed. Please check audit_and_fix.py and ensure:")
        print("  1. DATA_PATH points to your training data file")
        print("  2. The file has required columns (ctr_score, source, etc.)")
        print("\nExiting.")
        return
    
    # Check if cleaned datasets were created
    import glob
    exp1_files = glob.glob('data/*exp1*.csv')
    exp2_files = glob.glob('data/*exp2*.csv')
    exp3_files = glob.glob('data/*exp3*.csv')
    
    if not (exp1_files and exp2_files and exp3_files):
        print("\nERROR: Cleaned datasets not found. Please check audit_and_fix.py output.")
        return
    
    print(f"\nOK: Generated cleaned datasets:")
    print(f"  Exp1 (no sentinels): {exp1_files[0]}")
    print(f"  Exp2 (balanced): {exp2_files[0]}")
    print(f"  Exp3 (censored): {exp3_files[0]}")
    
    # Step 2: Train on each dataset
    experiments = [
        ('exp1', "Drop 960 sentinel rows (baseline)"),
        ('exp2', "Drop sentinels + balance to 60/40 real/synthetic"),
        ('exp3', "Keep sentinels as censored data"),
    ]
    
    results = {}
    
    for exp_name, exp_desc in experiments:
        print("\n" + "="*80)
        print(f"STEP 2.{exp_name[-1]}: {exp_desc}")
        print("="*80)
        
        # Run training
        cmd = f"py -3.11 retrain_experiments.py --dataset {exp_name} --epochs 30 --batch_size 64"
        
        if not run_command(cmd, f"Training model on {exp_name} dataset"):
            print(f"\nWARNING: Training failed for {exp_name}. Continuing with other experiments...")
            results[exp_name] = {'spearman_r': None, 'error': 'Training failed'}
            continue
        
        # Load results
        result_files = glob.glob(f'results_{exp_name}_*.json')
        if result_files:
            latest_result = sorted(result_files)[-1]
            with open(latest_result, 'r') as f:
                exp_results = json.load(f)
            results[exp_name] = exp_results['final_metrics']
            print(f"\nOK: {exp_name} completed: Spearman r = {results[exp_name]['spearman_r']:.4f}")
        else:
            print(f"\nWARNING: Could not find results file for {exp_name}")
            results[exp_name] = {'spearman_r': None, 'error': 'Results file not found'}
    
    # Step 3: Compare results
    print("\n" + "="*80)
    print("STEP 3: Results Comparison")
    print("="*80)
    
    print("\n=== Performance Summary ===\n")
    print(f"Baseline (original with sentinels): Spearman r = 0.254")
    print()
    
    comparison_data = []
    for exp_name, exp_desc in experiments:
        if results[exp_name]['spearman_r'] is not None:
            r = results[exp_name]['spearman_r']
            improvement = r - 0.254
            pct_improvement = 100 * improvement / 0.254
            
            print(f"{exp_name.upper()} ({exp_desc}):")
            print(f"  Spearman r: {r:.4f}")
            print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
            if 'auc' in results[exp_name]:
                print(f"  AUC: {results[exp_name]['auc']:.4f}")
            print()
            
            comparison_data.append({
                'Experiment': exp_name,
                'Description': exp_desc,
                'Spearman_r': r,
                'Improvement': improvement,
                'Improvement_pct': pct_improvement,
            })
        else:
            print(f"{exp_name.upper()}: FAILED")
            print()
    
    # Save comparison table
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        comparison_path = f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_comparison.to_csv(comparison_path, index=False)
        print(f"OK: Comparison saved to: {comparison_path}")
    
    # Step 4: Recommendations
    print("\n" + "="*80)
    print("STEP 4: Recommendations")
    print("="*80)
    
    if not comparison_data:
        print("\nNo successful experiments to compare. Please check error messages above.")
        return
    
    # Find best experiment
    best_exp = max(comparison_data, key=lambda x: x['Spearman_r'])
    
    print(f"\nBEST RESULT: {best_exp['Experiment'].upper()}")
    print(f"  Description: {best_exp['Description']}")
    print(f"  Spearman r: {best_exp['Spearman_r']:.4f}")
    print(f"  Improvement: {best_exp['Improvement']:+.4f} ({best_exp['Improvement_pct']:+.1f}%)")
    
    print(f"\n=== Analysis ===")
    
    # Assess if improvement is significant
    if best_exp['Spearman_r'] > 0.30:
        print(f"OK: Achieved r > 0.30 - This is a meaningful improvement!")
    elif best_exp['Spearman_r'] > 0.27:
        print(f"OK: Moderate improvement - data cleaning helped")
    else:
        print(f"WARNING: Limited improvement - sentinel removal had small impact")
    
    print(f"\n=== Q&A ===")
    print(f"\n1. Should you drop sentinel rows?")
    if best_exp['Improvement'] > 0.03:
        print(f"   YES - Removing sentinels improved Spearman r by {best_exp['Improvement']:.3f}")
        print(f"   Recommendation: Use {best_exp['Experiment']} dataset going forward")
    else:
        print(f"   MAYBE - Improvement was modest ({best_exp['Improvement']:.3f})")
        print(f"   The sentinels may not have been the main bottleneck")
    
    print(f"\n2. Should you downsample synthetic data?")
    exp2_result = next((x for x in comparison_data if x['Experiment'] == 'exp2'), None)
    exp1_result = next((x for x in comparison_data if x['Experiment'] == 'exp1'), None)
    
    if exp2_result and exp1_result:
        if exp2_result['Spearman_r'] > exp1_result['Spearman_r']:
            print(f"   YES - Balanced data (60/40) outperformed unbalanced")
            print(f"   Exp2 r={exp2_result['Spearman_r']:.3f} vs Exp1 r={exp1_result['Spearman_r']:.3f}")
        else:
            print(f"   NO - More synthetic data helped (or was neutral)")
            print(f"   Exp1 r={exp1_result['Spearman_r']:.3f} vs Exp2 r={exp2_result['Spearman_r']:.3f}")
    
    print(f"\n3. Censored data approach?")
    exp3_result = next((x for x in comparison_data if x['Experiment'] == 'exp3'), None)
    if exp3_result:
        if exp3_result['Spearman_r'] > best_exp['Spearman_r']:
            print(f"   YES - Treating sentinels as censored data worked best!")
            print(f"   This preserves information while handling missing CTR values")
        else:
            print(f"   NOT NECESSARY - Simpler approaches (drop sentinels) work as well")
    else:
        print(f"   NOT TESTED - Requires custom loss function implementation")
    
    print(f"\n4. Can you achieve Spearman r > 0.35?")
    if best_exp['Spearman_r'] >= 0.35:
        print(f"   YES - You already achieved r={best_exp['Spearman_r']:.3f}!")
    elif best_exp['Spearman_r'] >= 0.30:
        print(f"   POSSIBLY - Currently at r={best_exp['Spearman_r']:.3f}")
        print(f"   Next steps to reach 0.35+:")
        print(f"     - Hyperparameter tuning (learning rate, architecture)")
        print(f"     - Better features (add text, metadata)")
        print(f"     - Larger CLIP model (ViT-B/32 → ViT-L/14)")
        print(f"     - Collect more real data (currently {2542} clean apify rows)")
    else:
        print(f"   CHALLENGING - Currently at r={best_exp['Spearman_r']:.3f}")
        print(f"   The dataset size (~{2542 + 18746} samples) should support r≈0.4-0.5")
        print(f"   Gap suggests feature quality is the bottleneck, not just data cleaning")
        print(f"   Consider:")
        print(f"     - Adding text features (ad copy, headlines)")
        print(f"     - Including metadata (audience, placement, timing)")
        print(f"     - Ensemble models")
        print(f"     - Collecting more diverse real data")
    
    print(f"\n=== Next Steps ===")
    print(f"1. Use {best_exp['Experiment']} dataset for production")
    print(f"2. Save the best model: model_{best_exp['Experiment']}_best.pt")
    print(f"3. If target is r>0.35, focus on feature engineering (not just data cleaning)")
    
    print(f"\n{'='*80}")
    print(f"Experiments complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
