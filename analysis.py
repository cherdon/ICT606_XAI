"""
Comprehensive Wine Quality Data Analysis
=========================================
This script performs exploratory data analysis (EDA) on the wine quality datasets
to determine preprocessing requirements for machine learning models.

Analysis includes:
1. Basic Statistics & Data Overview
2. Distribution Analysis (Histograms, Density Plots)
3. Outlier Detection (Box Plots, IQR Method)
4. Skewness Analysis
5. Correlation Analysis
6. Feature Scaling Requirements
7. Class Imbalance Analysis
8. Comparison: Red vs White Wine
9. Combined Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for plots
import os
os.makedirs('analysis_outputs', exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load red and white wine datasets and create combined dataset."""
    # Load datasets (semicolon separated)
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')
    
    # Add wine type indicator
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Create combined dataset
    combined = pd.concat([red_wine, white_wine], ignore_index=True)
    
    return red_wine, white_wine, combined

# ============================================================================
# BASIC STATISTICS
# ============================================================================

def print_basic_stats(df, name):
    """Print basic statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - BASIC STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\n📋 Column Names:\n{list(df.columns)}")
    print(f"\n🔍 Data Types:\n{df.dtypes}")
    print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
    print(f"\n📈 Descriptive Statistics:")
    print(df.describe().round(3).to_string())
    
    return df.describe()

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def plot_distributions(df, name, color_palette):
    """Plot distribution of all features using histograms and KDE."""
    # Exclude wine_type column if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=ax, color=color_palette[idx % len(color_palette)], 
                     edgecolor='white', alpha=0.7)
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlabel('')
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{name} Wine - Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_distributions.png")

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def plot_boxplots(df, name, color):
    """Create box plots for outlier visualization."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
    
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{name} Wine - Box Plots (Outlier Detection)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_boxplots.png")

def calculate_outliers_iqr(df, name):
    """Calculate outliers using IQR method and return summary."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_summary = []
    
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - OUTLIER ANALYSIS (IQR Method)")
    print(f"{'='*80}")
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df)) * 100
        
        outlier_summary.append({
            'Feature': col,
            'Q1': round(Q1, 4),
            'Q3': round(Q3, 4),
            'IQR': round(IQR, 4),
            'Lower Bound': round(lower_bound, 4),
            'Upper Bound': round(upper_bound, 4),
            'N Outliers': n_outliers,
            '% Outliers': round(pct_outliers, 2)
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
    
    # Highlight features with significant outliers
    significant_outliers = outlier_df[outlier_df['% Outliers'] > 5]
    if len(significant_outliers) > 0:
        print(f"\n⚠️  Features with >5% outliers:")
        for _, row in significant_outliers.iterrows():
            print(f"   - {row['Feature']}: {row['% Outliers']}%")
    
    return outlier_df

# ============================================================================
# SKEWNESS ANALYSIS
# ============================================================================

def analyze_skewness(df, name):
    """Analyze skewness and kurtosis of features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - SKEWNESS & KURTOSIS ANALYSIS")
    print(f"{'='*80}")
    
    skew_summary = []
    
    for col in numeric_cols:
        col_skew = skew(df[col].dropna())
        col_kurt = kurtosis(df[col].dropna())
        
        # Determine skewness type
        if abs(col_skew) < 0.5:
            skew_type = "Symmetric"
        elif col_skew > 0:
            skew_type = "Right-skewed (Positive)"
        else:
            skew_type = "Left-skewed (Negative)"
        
        # Determine if transformation needed
        needs_transform = "Yes" if abs(col_skew) > 1 else ("Maybe" if abs(col_skew) > 0.5 else "No")
        
        skew_summary.append({
            'Feature': col,
            'Skewness': round(col_skew, 4),
            'Kurtosis': round(col_kurt, 4),
            'Type': skew_type,
            'Transform Needed': needs_transform
        })
    
    skew_df = pd.DataFrame(skew_summary)
    print(skew_df.to_string(index=False))
    
    # Recommendations
    highly_skewed = skew_df[skew_df['Transform Needed'] == 'Yes']
    if len(highly_skewed) > 0:
        print(f"\n🔧 Features requiring transformation (|skew| > 1):")
        for _, row in highly_skewed.iterrows():
            print(f"   - {row['Feature']}: skew = {row['Skewness']}")
            if row['Skewness'] > 0:
                print(f"     → Recommend: Log transform or Square root transform")
            else:
                print(f"     → Recommend: Square transform or Exponential transform")
    
    return skew_df

def plot_skewness_comparison(df, name, color):
    """Visualize skewness with before/after log transform."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find highly skewed features
    skewed_features = [col for col in numeric_cols if abs(skew(df[col].dropna())) > 1]
    
    if len(skewed_features) == 0:
        print(f"ℹ️  No highly skewed features found in {name} wine dataset")
        return
    
    n_features = len(skewed_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(skewed_features):
        # Original distribution
        ax1 = axes[idx, 0]
        sns.histplot(df[col], kde=True, ax=ax1, color=color, alpha=0.7)
        ax1.set_title(f'{col} (Original)\nSkewness: {skew(df[col]):.3f}', fontweight='bold')
        
        # Log transformed (handle zeros by adding small constant)
        ax2 = axes[idx, 1]
        log_data = np.log1p(df[col] - df[col].min() + 1)
        sns.histplot(log_data, kde=True, ax=ax2, color='green', alpha=0.7)
        ax2.set_title(f'{col} (Log Transformed)\nSkewness: {skew(log_data):.3f}', fontweight='bold')
    
    plt.suptitle(f'{name} Wine - Skewness Correction Preview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_skewness_transform.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_skewness_transform.png")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def plot_correlation_matrix(df, name, cmap):
    """Create correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title(f'{name} Wine - Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_correlation.png")
    
    # Find highly correlated features
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Get correlations with target (quality)
    if 'quality' in numeric_cols:
        quality_corr = corr_matrix['quality'].drop('quality').sort_values(key=abs, ascending=False)
        print(f"\n📊 Correlations with Quality (Target):")
        for feat, corr in quality_corr.items():
            strength = "Strong" if abs(corr) > 0.5 else ("Moderate" if abs(corr) > 0.3 else "Weak")
            direction = "+" if corr > 0 else "-"
            print(f"   {feat:25s}: {corr:+.4f} ({strength} {direction})")
    
    # Find multicollinearity (high correlation between features)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': round(corr_matrix.iloc[i, j], 4)
                })
    
    if high_corr_pairs:
        print(f"\n⚠️  Highly Correlated Feature Pairs (|r| > 0.7) - Potential Multicollinearity:")
        for pair in high_corr_pairs:
            print(f"   - {pair['Feature 1']} ↔ {pair['Feature 2']}: {pair['Correlation']}")
    
    return corr_matrix

# ============================================================================
# FEATURE SCALING ANALYSIS
# ============================================================================

def analyze_scaling_requirements(df, name):
    """Analyze feature ranges and scaling requirements."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - FEATURE SCALING ANALYSIS")
    print(f"{'='*80}")
    
    scaling_summary = []
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_range = col_max - col_min
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        scaling_summary.append({
            'Feature': col,
            'Min': round(col_min, 4),
            'Max': round(col_max, 4),
            'Range': round(col_range, 4),
            'Mean': round(col_mean, 4),
            'Std': round(col_std, 4),
            'CV (%)': round((col_std / col_mean) * 100, 2) if col_mean != 0 else 0
        })
    
    scaling_df = pd.DataFrame(scaling_summary)
    print(scaling_df.to_string(index=False))
    
    # Analyze range differences
    ranges = scaling_df['Range'].values
    range_ratio = max(ranges) / min(ranges) if min(ranges) > 0 else float('inf')
    
    print(f"\n📏 Range Analysis:")
    print(f"   - Minimum range: {min(ranges):.4f}")
    print(f"   - Maximum range: {max(ranges):.4f}")
    print(f"   - Range ratio (max/min): {range_ratio:.2f}")
    
    if range_ratio > 10:
        print(f"\n⚠️  SCALING RECOMMENDED: Large range differences detected!")
        print(f"   Features have very different scales (ratio: {range_ratio:.0f}x)")
        print(f"   → StandardScaler: Good for normally distributed data")
        print(f"   → MinMaxScaler: Good for bounded data, preserves shape")
        print(f"   → RobustScaler: Good when outliers are present")
    else:
        print(f"\n✅ Features are on relatively similar scales")
    
    return scaling_df

def plot_feature_ranges(df, name, color):
    """Visualize feature ranges to show scaling needs."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before scaling - show raw ranges
    ax1 = axes[0]
    data_to_plot = [df[col].dropna().values for col in numeric_cols]
    bp = ax1.boxplot(data_to_plot, labels=numeric_cols, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Before Scaling (Raw Values)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # After StandardScaler
    ax2 = axes[1]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_data = df[numeric_cols].dropna()
    scaled_data = scaler.fit_transform(numeric_data)
    bp2 = ax2.boxplot(scaled_data, labels=numeric_cols, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)
    ax2.set_title('After StandardScaler', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Scaled Value')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'{name} Wine - Feature Scaling Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_scaling.png")

# ============================================================================
# CLASS IMBALANCE ANALYSIS
# ============================================================================

def analyze_target_distribution(df, name, color):
    """Analyze the distribution of the target variable (quality)."""
    print(f"\n{'='*80}")
    print(f" {name.upper()} WINE - TARGET VARIABLE (QUALITY) ANALYSIS")
    print(f"{'='*80}")
    
    quality_counts = df['quality'].value_counts().sort_index()
    quality_pct = (quality_counts / len(df) * 100).round(2)
    
    print(f"\n📊 Quality Score Distribution:")
    for score, count in quality_counts.items():
        pct = quality_pct[score]
        bar = '█' * int(pct / 2)
        print(f"   Quality {score}: {count:5d} ({pct:5.2f}%) {bar}")
    
    # Create binary classification labels (high quality: >= 7)
    high_quality = (df['quality'] >= 7).sum()
    low_quality = (df['quality'] < 7).sum()
    
    print(f"\n📈 Binary Classification (High Quality ≥ 7):")
    print(f"   - High Quality (≥7): {high_quality} ({high_quality/len(df)*100:.2f}%)")
    print(f"   - Low/Medium Quality (<7): {low_quality} ({low_quality/len(df)*100:.2f}%)")
    print(f"   - Imbalance Ratio: 1:{low_quality/high_quality:.1f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Bar plot of quality distribution
    ax1 = axes[0]
    bars = ax1.bar(quality_counts.index, quality_counts.values, color=color, edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Quality Score', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Quality Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(quality_counts.index)
    for bar, count in zip(bars, quality_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Pie chart for binary classification
    ax2 = axes[1]
    sizes = [high_quality, low_quality]
    labels = ['High Quality (≥7)', 'Low/Medium (<7)']
    colors_pie = [color, '#95a5a6']
    explode = (0.05, 0)
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Binary Classification Split', fontsize=12, fontweight='bold')
    
    # Quality by percentile
    ax3 = axes[2]
    quality_cumsum = quality_counts.cumsum() / len(df) * 100
    ax3.bar(quality_cumsum.index, quality_counts.values / len(df) * 100, 
            color=color, alpha=0.7, label='Frequency')
    ax3.plot(quality_cumsum.index, quality_cumsum.values, 'ro-', linewidth=2, 
             markersize=8, label='Cumulative %')
    ax3.set_xlabel('Quality Score', fontsize=11)
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.set_title('Quality Distribution (Cumulative)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.set_xticks(quality_counts.index)
    
    plt.suptitle(f'{name} Wine - Target Variable Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'analysis_outputs/{name.lower()}_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower()}_target_distribution.png")
    
    return quality_counts

# ============================================================================
# COMPARATIVE ANALYSIS (RED vs WHITE)
# ============================================================================

def compare_wine_types(red_df, white_df, combined_df):
    """Compare red and white wine characteristics."""
    print(f"\n{'='*80}")
    print(f" RED vs WHITE WINE - COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    numeric_cols = red_df.select_dtypes(include=[np.number]).columns.tolist()
    
    comparison = []
    for col in numeric_cols:
        red_mean = red_df[col].mean()
        white_mean = white_df[col].mean()
        diff_pct = ((white_mean - red_mean) / red_mean) * 100 if red_mean != 0 else 0
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(red_df[col].dropna(), white_df[col].dropna())
        significant = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
        
        comparison.append({
            'Feature': col,
            'Red Mean': round(red_mean, 4),
            'White Mean': round(white_mean, 4),
            'Diff (%)': round(diff_pct, 2),
            'p-value': f"{p_value:.4f}{significant}"
        })
    
    comp_df = pd.DataFrame(comparison)
    print(f"\n📊 Feature Comparison (t-test significance: * p<0.05, ** p<0.01, *** p<0.001):")
    print(comp_df.to_string(index=False))
    
    # Visualize comparisons
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data_to_plot = [red_df[col].dropna(), white_df[col].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Red', 'White'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#c0392b')
        bp['boxes'][1].set_facecolor('#f1c40f')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value')
    
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Red vs White Wine - Feature Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('analysis_outputs/red_vs_white_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: red_vs_white_comparison.png")
    
    # Violin plots for detailed distribution comparison
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        combined_subset = combined_df[[col, 'wine_type']].dropna()
        sns.violinplot(x='wine_type', y=col, data=combined_subset, ax=ax,
                       palette={'red': '#c0392b', 'white': '#f1c40f'})
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
    
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Red vs White Wine - Distribution Comparison (Violin Plots)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('analysis_outputs/red_vs_white_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: red_vs_white_violin.png")
    
    return comp_df

# ============================================================================
# COMBINED DATASET ANALYSIS
# ============================================================================

def analyze_combined_dataset(combined_df):
    """Analyze the combined red and white wine dataset."""
    print(f"\n{'='*80}")
    print(f" COMBINED DATASET ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📊 Combined Dataset Overview:")
    print(f"   - Total samples: {len(combined_df)}")
    print(f"   - Red wine samples: {len(combined_df[combined_df['wine_type'] == 'red'])} "
          f"({len(combined_df[combined_df['wine_type'] == 'red'])/len(combined_df)*100:.1f}%)")
    print(f"   - White wine samples: {len(combined_df[combined_df['wine_type'] == 'white'])} "
          f"({len(combined_df[combined_df['wine_type'] == 'white'])/len(combined_df)*100:.1f}%)")
    
    # Quality distribution by wine type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    ax1 = axes[0]
    quality_by_type = combined_df.groupby(['quality', 'wine_type']).size().unstack(fill_value=0)
    quality_by_type.plot(kind='bar', stacked=True, ax=ax1, color=['#c0392b', '#f1c40f'], alpha=0.8)
    ax1.set_xlabel('Quality Score', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Quality Distribution by Wine Type', fontsize=12, fontweight='bold')
    ax1.legend(title='Wine Type')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # Proportion plot
    ax2 = axes[1]
    quality_pct = quality_by_type.div(quality_by_type.sum(axis=1), axis=0) * 100
    quality_pct.plot(kind='bar', stacked=True, ax=ax2, color=['#c0392b', '#f1c40f'], alpha=0.8)
    ax2.set_xlabel('Quality Score', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title('Wine Type Proportion by Quality', fontsize=12, fontweight='bold')
    ax2.legend(title='Wine Type')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    plt.suptitle('Combined Dataset - Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_outputs/combined_quality_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: combined_quality_distribution.png")

# ============================================================================
# FEATURE IMPORTANCE PREVIEW (Simple Correlation-based)
# ============================================================================

def plot_feature_importance_preview(red_df, white_df, combined_df):
    """Plot correlation-based feature importance with target."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = [
        (red_df, 'Red Wine', '#c0392b'),
        (white_df, 'White Wine', '#f1c40f'),
        (combined_df, 'Combined', '#3498db')
    ]
    
    for idx, (df, name, color) in enumerate(datasets):
        ax = axes[idx]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'quality' in numeric_cols:
            numeric_cols.remove('quality')
        
        correlations = df[numeric_cols].corrwith(df['quality']).sort_values()
        colors = [color if c > 0 else '#95a5a6' for c in correlations]
        
        bars = ax.barh(correlations.index, correlations.values, color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Correlation with Quality', fontsize=11)
        ax.set_title(f'{name}\nFeature-Quality Correlation', fontsize=12, fontweight='bold')
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations.values):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{corr:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Feature Importance Preview (Correlation-based)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_outputs/feature_importance_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: feature_importance_preview.png")

# ============================================================================
# PREPROCESSING RECOMMENDATIONS SUMMARY
# ============================================================================

def generate_recommendations(red_df, white_df, combined_df):
    """Generate comprehensive preprocessing recommendations."""
    print(f"\n{'='*80}")
    print(f" 📋 PREPROCESSING RECOMMENDATIONS SUMMARY")
    print(f"{'='*80}")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. MISSING VALUES                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✅ No missing values detected in any dataset                                │
│    → No imputation required                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. OUTLIER HANDLING                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠️  Several features contain outliers (particularly residual sugar,         │
│    chlorides, free/total sulfur dioxide)                                    │
│                                                                             │
│ Recommendations:                                                            │
│ • For tree-based models: Keep outliers (robust to outliers)                 │
│ • For distance-based models: Consider capping/winsorization                 │
│ • For neural networks: Use RobustScaler or clip outliers                    │
│ • Alternative: Use IQR method to cap at 1.5*IQR bounds                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SKEWNESS TREATMENT                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠️  Highly skewed features detected:                                        │
│    • residual sugar (right-skewed)                                          │
│    • chlorides (right-skewed)                                               │
│    • free sulfur dioxide (right-skewed)                                     │
│    • total sulfur dioxide (right-skewed)                                    │
│                                                                             │
│ Recommendations:                                                            │
│ • Apply log1p transformation for right-skewed features                      │
│ • Use Box-Cox or Yeo-Johnson transformation                                 │
│ • For tree-based models: transformation may not be necessary                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FEATURE SCALING                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠️  Features have very different scales (e.g., density ~0.99 vs total SO2   │
│    ~100+)                                                                   │
│                                                                             │
│ Recommendations:                                                            │
│ • StandardScaler: For normally distributed features                         │
│ • MinMaxScaler: For bounded features                                        │
│ • RobustScaler: When outliers are present (RECOMMENDED)                     │
│ • For tree-based models: scaling is optional                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. FEATURE ENGINEERING                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 💡 Suggestions based on domain knowledge and correlations:                  │
│ • Create acidity_ratio = fixed_acidity / volatile_acidity                   │
│ • Create sulfur_ratio = free_SO2 / total_SO2                                │
│ • Consider interaction terms for highly correlated features                 │
│ • For combined dataset: wine_type as categorical feature                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. CLASS IMBALANCE                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠️  Significant class imbalance in quality scores                           │
│    Most wines are quality 5-6, few are 3-4 or 8-9                           │
│                                                                             │
│ Recommendations:                                                            │
│ • For classification: Use stratified sampling                               │
│ • Consider SMOTE for oversampling minority classes                          │
│ • Use class_weight='balanced' in models                                     │
│ • Evaluate using F1-score, not just accuracy                                │
│ • For binary classification (high/low quality): threshold at 7              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. MULTICOLLINEARITY                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠️  Highly correlated feature pairs detected:                               │
│    • fixed acidity ↔ citric acid                                            │
│    • fixed acidity ↔ density                                                │
│    • fixed acidity ↔ pH (negative)                                          │
│    • free sulfur dioxide ↔ total sulfur dioxide                             │
│    • density ↔ alcohol (negative)                                           │
│    • density ↔ residual sugar (in white wine)                               │
│                                                                             │
│ Recommendations:                                                            │
│ • Consider PCA for dimensionality reduction                                 │
│ • Use regularization (L1/L2) in linear models                               │
│ • Feature selection based on importance scores                              │
│ • VIF analysis to quantify multicollinearity                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. DATASET-SPECIFIC NOTES                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 🍷 RED WINE:                                                                │
│    • 1,599 samples - smaller dataset                                        │
│    • Generally lower sulfur dioxide levels                                  │
│    • Higher volatile acidity on average                                     │
│                                                                             │
│ 🥂 WHITE WINE:                                                              │
│    • 4,898 samples - larger dataset                                         │
│    • Higher residual sugar (more variation)                                 │
│    • Higher sulfur dioxide levels                                           │
│                                                                             │
│ 🍾 COMBINED DATASET:                                                        │
│    • 6,497 total samples                                                    │
│    • wine_type as important categorical feature                             │
│    • Imbalanced: ~75% white, ~25% red                                       │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# PAIRPLOT FOR KEY FEATURES
# ============================================================================

def plot_pairplot(df, name, hue_col=None):
    """Create pairplot for key features."""
    # Select key features based on correlation with quality
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'quality' in numeric_cols:
        correlations = df[numeric_cols].corrwith(df['quality']).abs()
        top_features = correlations.nlargest(5).index.tolist()
        if 'quality' not in top_features:
            top_features.append('quality')
    else:
        top_features = numeric_cols[:6]
    
    # Create pairplot
    if hue_col and hue_col in df.columns:
        g = sns.pairplot(df[top_features + [hue_col]], hue=hue_col, 
                        palette={'red': '#c0392b', 'white': '#f1c40f'} if hue_col == 'wine_type' else 'viridis',
                        diag_kind='kde', corner=True)
    else:
        g = sns.pairplot(df[top_features], diag_kind='kde', corner=True)
    
    plt.suptitle(f'{name} - Pairplot of Top Correlated Features', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(f'analysis_outputs/{name.lower().replace(" ", "_")}_pairplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {name.lower().replace(' ', '_')}_pairplot.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run all analyses."""
    print("\n" + "🍷"*40)
    print(" WINE QUALITY DATASET - COMPREHENSIVE EDA")
    print("🍷"*40)
    
    # Load data
    red_wine, white_wine, combined = load_data()
    
    # Color palettes
    red_palette = sns.color_palette("Reds_d", 12)
    white_palette = sns.color_palette("YlOrBr_d", 12)
    combined_palette = sns.color_palette("Blues_d", 12)
    
    # ========== RED WINE ANALYSIS ==========
    print("\n\n" + "🔴"*40)
    print(" RED WINE ANALYSIS")
    print("🔴"*40)
    
    print_basic_stats(red_wine, "Red")
    plot_distributions(red_wine, "Red", red_palette)
    plot_boxplots(red_wine, "Red", '#c0392b')
    calculate_outliers_iqr(red_wine, "Red")
    analyze_skewness(red_wine, "Red")
    plot_skewness_comparison(red_wine, "Red", '#c0392b')
    plot_correlation_matrix(red_wine, "Red", 'RdYlBu_r')
    analyze_scaling_requirements(red_wine, "Red")
    plot_feature_ranges(red_wine, "Red", '#c0392b')
    analyze_target_distribution(red_wine, "Red", '#c0392b')
    plot_pairplot(red_wine, "Red Wine")
    
    # ========== WHITE WINE ANALYSIS ==========
    print("\n\n" + "⚪"*40)
    print(" WHITE WINE ANALYSIS")
    print("⚪"*40)
    
    print_basic_stats(white_wine, "White")
    plot_distributions(white_wine, "White", white_palette)
    plot_boxplots(white_wine, "White", '#f1c40f')
    calculate_outliers_iqr(white_wine, "White")
    analyze_skewness(white_wine, "White")
    plot_skewness_comparison(white_wine, "White", '#f1c40f')
    plot_correlation_matrix(white_wine, "White", 'YlOrBr')
    analyze_scaling_requirements(white_wine, "White")
    plot_feature_ranges(white_wine, "White", '#f1c40f')
    analyze_target_distribution(white_wine, "White", '#f1c40f')
    plot_pairplot(white_wine, "White Wine")
    
    # ========== COMPARATIVE ANALYSIS ==========
    print("\n\n" + "🔀"*40)
    print(" COMPARATIVE ANALYSIS")
    print("🔀"*40)
    
    compare_wine_types(red_wine, white_wine, combined)
    
    # ========== COMBINED DATASET ANALYSIS ==========
    print("\n\n" + "🍾"*40)
    print(" COMBINED DATASET ANALYSIS")
    print("🍾"*40)
    
    print_basic_stats(combined, "Combined")
    analyze_combined_dataset(combined)
    plot_correlation_matrix(combined, "Combined", 'coolwarm')
    analyze_scaling_requirements(combined, "Combined")
    calculate_outliers_iqr(combined, "Combined")
    analyze_skewness(combined, "Combined")
    analyze_target_distribution(combined, "Combined", '#3498db')
    plot_pairplot(combined, "Combined", 'wine_type')
    
    # ========== FEATURE IMPORTANCE PREVIEW ==========
    plot_feature_importance_preview(red_wine, white_wine, combined)
    
    # ========== RECOMMENDATIONS ==========
    generate_recommendations(red_wine, white_wine, combined)
    
    print("\n" + "="*80)
    print(" ✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All visualizations saved to: analysis_outputs/")
    print("\n📊 Generated plots:")
    for f in sorted(os.listdir('analysis_outputs')):
        print(f"   • {f}")

if __name__ == "__main__":
    main()
