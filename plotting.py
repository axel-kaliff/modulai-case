import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker


# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
COLORS = {
    'categorical': '#4C72B0',
    'text': '#55A868', 
    'image': '#C44E52',
    'fusion': '#8172B3',
    'baseline': '#937860',
    'best': '#CCB974',
}

MODALITY_COLORS = {
    'Cat/Num': COLORS['categorical'],
    'Text': COLORS['text'],
    'Image': COLORS['image'],
    'Stacking': COLORS['fusion'],
    'Early Fusion': COLORS['fusion'],
    'Simple Average': COLORS['baseline'],
}


def get_model_color(model_name):
    """Assign colors based on model/modality type."""
    for key, color in MODALITY_COLORS.items():
        if key in model_name:
            return color
    return '#666666'


def plot_model_comparison_dashboard(all_results, save_path=None):
    """
    Create a comprehensive dashboard comparing all models.
    """
    df = pd.DataFrame(all_results)
    df = df.sort_values('test_rmse', ascending=True).reset_index(drop=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    colors = [get_model_color(name) for name in df['model_name']]
    
    # --- Main comparison: RMSE with ranking ---
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.barh(df['model_name'], df['test_rmse'], color=colors, edgecolor='white', linewidth=1.5)
    
    # Highlight best model
    best_idx = df['test_rmse'].idxmin()
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(3)
    
    ax1.set_xlabel('Test RMSE (lower is better)')
    ax1.set_title('Model Performance Ranking')
    ax1.invert_yaxis()
    
    # Add value labels with ranking
    for i, (bar, val) in enumerate(zip(bars, df['test_rmse'])):
        rank = i + 1
        badge = f'#{rank}'
        weight = 'bold' if rank <= 3 else 'normal'
        color = '#FFD700' if rank == 1 else ('#C0C0C0' if rank == 2 else ('#CD7F32' if rank == 3 else 'black'))
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{badge} {val:.4f}', va='center', fontsize=10, fontweight=weight, color=color)
    
    ax1.set_xlim(0, df['test_rmse'].max() * 1.15)
    
    # --- R² comparison ---
    ax2 = fig.add_subplot(gs[0, 2])
    df_sorted_r2 = df.sort_values('test_r2', ascending=True)
    colors_r2 = [get_model_color(name) for name in df_sorted_r2['model_name']]
    
    bars_r2 = ax2.barh(df_sorted_r2['model_name'], df_sorted_r2['test_r2'], color=colors_r2, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Test R² (higher is better)')
    ax2.set_title('Explained Variance')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars_r2, df_sorted_r2['test_r2']):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # --- Metric comparison table ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    table_data = df[['model_name', 'test_rmse', 'test_mae', 'test_r2']].copy()
    table_data.columns = ['Model', 'RMSE', 'MAE', 'R²']
    table_data['RMSE'] = table_data['RMSE'].apply(lambda x: f'{x:.4f}')
    table_data['MAE'] = table_data['MAE'].apply(lambda x: f'{x:.4f}')
    table_data['R²'] = table_data['R²'].apply(lambda x: f'{x:.4f}')
    
    table = ax3.table(cellText=table_data.values,
                      colLabels=table_data.columns,
                      loc='center',
                      cellLoc='center',
                      colColours=['#E8E8E8']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Highlight best row
    for j in range(4):
        table[(1, j)].set_facecolor('#E8F5E9')
    
    ax3.set_title('Performance Summary', pad=20)
    
    # --- Improvement over baseline ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate improvement over worst single-modality model
    single_modality = df[df['model_name'].str.contains('Cat/Num|Text|Image', regex=True)]
    baseline_rmse = single_modality['test_rmse'].max()
    
    improvements = ((baseline_rmse - df['test_rmse']) / baseline_rmse * 100)
    colors_imp = ['#2E7D32' if imp > 0 else '#C62828' for imp in improvements]
    
    bars_imp = ax4.barh(df['model_name'], improvements, color=colors_imp, edgecolor='white', linewidth=1)
    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax4.set_xlabel('RMSE Improvement (%)')
    ax4.set_title(f'Improvement vs Worst Single-Modality')
    ax4.invert_yaxis()
    
    for bar, val in zip(bars_imp, improvements):
        offset = 0.3 if val >= 0 else -0.3
        ha = 'left' if val >= 0 else 'right'
        ax4.text(val + offset, bar.get_y() + bar.get_height()/2, 
                f'{val:+.1f}%', va='center', ha=ha, fontsize=9)
    
    # --- Legend ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    legend_elements = [
        Patch(facecolor=COLORS['categorical'], label='Categorical/Numerical'),
        Patch(facecolor=COLORS['text'], label='Text Embeddings'),
        Patch(facecolor=COLORS['image'], label='Image Embeddings'),
        Patch(facecolor=COLORS['fusion'], label='Multi-Modal Fusion'),
        Patch(facecolor=COLORS['baseline'], label='Baseline'),
    ]
    ax5.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True,
               title='Modality Types', title_fontsize=12)
    
    fig.suptitle('Multi-Modal Movie Rating Prediction: Model Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved dashboard to {save_path}")
    
    plt.show()
    return fig


def plot_predictions_analysis(y_test, y_pred, model_name, save_path=None):
    """
    Create detailed prediction analysis with density visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    residuals = y_test - y_pred
    
    # --- Scatter with density ---
    ax1 = axes[0, 0]
    hb = ax1.hexbin(y_test, y_pred, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax1.plot([1, 5], [1, 5], 'k--', lw=2, label='Perfect prediction', alpha=0.7)
    ax1.set_xlabel('Actual Rating')
    ax1.set_ylabel('Predicted Rating')
    ax1.set_title('Predicted vs Actual (Density)')
    ax1.legend(loc='upper left')
    ax1.set_xlim(1, 5)
    ax1.set_ylim(1, 5)
    ax1.set_aspect('equal')
    cb = plt.colorbar(hb, ax=ax1)
    cb.set_label('Count')
    
    # Add R² and RMSE annotation
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Residual distribution ---
    ax2 = axes[0, 1]
    sns.histplot(residuals, kde=True, ax=ax2, color=COLORS['fusion'], edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero error')
    ax2.axvline(x=residuals.mean(), color='orange', linestyle='-', lw=2, label=f'Mean: {residuals.mean():.3f}')
    ax2.set_xlabel('Residual (Actual - Predicted)')
    ax2.set_ylabel('Count')
    ax2.set_title('Residual Distribution')
    ax2.legend()
    
    # Add stats
    ax2.text(0.95, 0.95, f'Std: {residuals.std():.3f}\nSkew: {pd.Series(residuals).skew():.3f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Residuals vs Predicted ---
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.3, s=15, c=COLORS['categorical'])
    ax3.axhline(y=0, color='red', linestyle='--', lw=2)
    
    # Add smoothed trend line
    from scipy.ndimage import uniform_filter1d
    sorted_idx = np.argsort(y_pred)
    window = max(len(y_pred) // 20, 10)
    smoothed = uniform_filter1d(residuals[sorted_idx], size=window)
    ax3.plot(y_pred[sorted_idx], smoothed, color='orange', lw=2, label='Trend')
    
    ax3.set_xlabel('Predicted Rating')
    ax3.set_ylabel('Residual')
    ax3.set_title('Residuals vs Predicted (Heteroscedasticity Check)')
    ax3.legend()
    
    # Add ±1 std bands
    ax3.fill_between([y_pred.min(), y_pred.max()], 
                     [-residuals.std(), -residuals.std()],
                     [residuals.std(), residuals.std()],
                     alpha=0.1, color='gray', label='±1 std')
    
    # --- Error by rating range ---
    ax4 = axes[1, 1]
    
    bins = [1, 2, 2.5, 3, 3.5, 4, 5]
    bin_labels = ['1-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-5']
    y_binned = pd.cut(y_test, bins=bins, labels=bin_labels, include_lowest=True)
    
    error_by_bin = pd.DataFrame({'bin': y_binned, 'abs_error': np.abs(residuals)})
    error_stats = error_by_bin.groupby('bin')['abs_error'].agg(['mean', 'std', 'count'])
    
    bars = ax4.bar(error_stats.index, error_stats['mean'], 
                   yerr=error_stats['std'], capsize=5,
                   color=COLORS['text'], edgecolor='white', linewidth=1.5, alpha=0.8)
    
    # Add count labels
    for bar, count in zip(bars, error_stats['count']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={int(count)}', ha='center', fontsize=9, color='gray')
    
    ax4.set_xlabel('Actual Rating Range')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Prediction Error by Rating Range')
    
    fig.suptitle(f'{model_name}: Prediction Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved prediction analysis to {save_path}")
    
    plt.show()
    return fig


def plot_modality_contribution(modality_preds, y_test, weights, save_path=None):
    """
    Visualize how each modality contributes to the final prediction.
    """
    pred_cat, pred_text, pred_image = modality_preds
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    modalities = [
        ('Categorical/Numerical', pred_cat, COLORS['categorical']),
        ('Text Embeddings', pred_text, COLORS['text']),
        ('Image Embeddings', pred_image, COLORS['image']),
    ]
    
    for ax, (name, pred, color), weight in zip(axes, modalities, weights):
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        ax.hexbin(y_test, pred, gridsize=25, cmap='Blues', mincnt=1)
        ax.plot([1, 5], [1, 5], 'r--', lw=2, alpha=0.7)
        ax.set_xlabel('Actual Rating')
        ax.set_ylabel('Predicted Rating')
        ax.set_title(f'{name}\nWeight: {weight:.3f}')
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.set_aspect('equal')
        
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Individual Modality Predictions & Stacking Weights', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved modality contribution plot to {save_path}")
    
    plt.show()
    return fig


def plot_cv_vs_test_performance(all_results, save_path=None):
    """
    Compare cross-validation vs test performance to check for overfitting.
    """
    df = pd.DataFrame(all_results)
    df = df[df['cv_rmse'].notna()].copy()  # Filter out models without CV
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = [get_model_color(name) for name in df['model_name']]
    
    # RMSE comparison
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['cv_rmse'], width, label='CV RMSE', color=colors, alpha=0.7, edgecolor='white')
    bars2 = ax1.bar(x + width/2, df['test_rmse'], width, label='Test RMSE', color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Cross-Validation vs Test RMSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax1.legend()
    
    # Add difference annotations
    for i, (cv, test) in enumerate(zip(df['cv_rmse'], df['test_rmse'])):
        diff = test - cv
        color = '#C62828' if diff > 0.01 else '#2E7D32'
        ax1.annotate(f'{diff:+.3f}', xy=(i, max(cv, test) + 0.01), ha='center', fontsize=9, color=color)
    
    # R² comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, df['cv_r2'], width, label='CV R²', color=colors, alpha=0.7, edgecolor='white')
    bars4 = ax2.bar(x + width/2, df['test_r2'], width, label='Test R²', color=colors, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R²')
    ax2.set_title('Cross-Validation vs Test R²')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax2.legend()
    
    fig.suptitle('Overfitting Analysis: CV vs Test Performance', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved CV vs test plot to {save_path}")
    
    plt.show()
    return fig


def create_summary_slide(all_results, best_model_name, best_y_pred, y_test, save_path=None):
    """
    Create a single-slide summary suitable for presentations.
    """
    df = pd.DataFrame(all_results).sort_values('test_rmse')
    best = df.iloc[0]
    
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
    
    # --- Title area ---
    fig.suptitle('Multi-Modal Movie Rating Prediction\nKey Results', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # --- Key metrics boxes ---
    metrics = [
        ('Best RMSE', f"{best['test_rmse']:.4f}", COLORS['fusion']),
        ('Best R²', f"{best['test_r2']:.4f}", COLORS['text']),
        ('Best Model', best['model_name'], COLORS['categorical']),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, facecolor=color, alpha=0.2, edgecolor=color, linewidth=3))
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=18, fontweight='bold')
        ax.text(0.5, 0.3, label, ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
    
    # --- Improvement stat ---
    ax_imp = fig.add_subplot(gs[0, 3])
    single_best = df[df['model_name'].str.contains('Cat/Num|Text|Image', regex=True)]['test_rmse'].min()
    improvement = (single_best - best['test_rmse']) / single_best * 100
    
    ax_imp.set_xlim(0, 1)
    ax_imp.set_ylim(0, 1)
    color = '#2E7D32' if improvement > 0 else '#C62828'
    ax_imp.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, facecolor=color, alpha=0.2, edgecolor=color, linewidth=3))
    ax_imp.text(0.5, 0.65, f"+{improvement:.1f}%", ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    ax_imp.text(0.5, 0.3, 'Fusion Gain', ha='center', va='center', fontsize=12, color='gray')
    ax_imp.axis('off')
    
    # --- Model ranking bar chart ---
    ax_rank = fig.add_subplot(gs[1, :2])
    colors = [get_model_color(name) for name in df['model_name']]
    bars = ax_rank.barh(df['model_name'], df['test_rmse'], color=colors, edgecolor='white', linewidth=1.5)
    bars[0].set_edgecolor('#FFD700')
    bars[0].set_linewidth(3)
    ax_rank.set_xlabel('Test RMSE (lower is better)', fontsize=11)
    ax_rank.set_title('Model Performance Ranking', fontsize=13, fontweight='bold')
    ax_rank.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, df['test_rmse'])):
        ax_rank.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)
    
    # --- Prediction scatter ---
    ax_pred = fig.add_subplot(gs[1, 2:])
    residuals = y_test - best_y_pred
    
    hb = ax_pred.hexbin(y_test, best_y_pred, gridsize=25, cmap='YlOrRd', mincnt=1)
    ax_pred.plot([1, 5], [1, 5], 'k--', lw=2, alpha=0.7)
    ax_pred.set_xlabel('Actual Rating', fontsize=11)
    ax_pred.set_ylabel('Predicted Rating', fontsize=11)
    ax_pred.set_title(f'{best_model_name}: Predictions', fontsize=13, fontweight='bold')
    ax_pred.set_xlim(1, 5)
    ax_pred.set_ylim(1, 5)
    ax_pred.set_aspect('equal')
    plt.colorbar(hb, ax=ax_pred, label='Count')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved summary slide to {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Replacement for original functions in solution.py
# ============================================================

def plot_results_comparison(all_results, save_path=None):
    """Enhanced version of original function."""
    return plot_model_comparison_dashboard(all_results, save_path)


def plot_predictions_vs_actual(y_test, y_pred, model_name, save_path=None):
    """Enhanced version of original function."""
    return plot_predictions_analysis(y_test, y_pred, model_name, save_path)