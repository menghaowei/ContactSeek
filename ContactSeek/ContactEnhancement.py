# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime

from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =========================== RF Importance Calculation ===========================
def calculate_residue_rf_importance(models, keep_residues,
                                    cp_raw_feature_count=18, cp_diff_feature_count=6):
    """
    Calculate Random Forest feature importance for each retained residue.
    
    This function aggregates feature importance scores across all features 
    belonging to each residue, then averages across multiple RF models.
    
    Parameters
    ----------
    models : list of dict
        List of trained Random Forest model dictionaries. Each dict must contain:
        - 'model': sklearn.ensemble.RandomForestClassifier with feature_importances_
    keep_residues : numpy.ndarray
        Boolean array of shape (n_total_residues,) indicating which residues 
        were retained for training (True = kept, False = filtered).
    cp_raw_feature_count : int, default=18
        Number of raw CP features per residue (e.g. 18 for Cas9, 9 for A3A, 3 for TadA8e).
    cp_diff_feature_count : int, default=6
        Number of diff CP features per residue (e.g. 6 for Cas9, 0 for A3A/TadA8e).
    
    Returns
    -------
    residue_importance_dict : dict
        Dictionary mapping residue position (0-based) to importance info:
        - 'position': int, 1-based residue position
        - 'rf_score': float, mean RF importance across all models
        - 'model_scores': list of float, individual model scores
    
    Notes
    -----
    Feature Organization:
    - Each residue has (cp_raw_feature_count + cp_diff_feature_count) features
    - Feature indices are organized as:
      [Res1_raw, Res2_raw, ..., ResN_raw, Res1_diff, ..., ResN_diff]
    - For residue i (in kept_residues):
      - Raw features: indices [i*cp_raw : (i+1)*cp_raw]
      - Diff features: indices [N*cp_raw + i*cp_diff : N*cp_raw + (i+1)*cp_diff]
      - If cp_diff_feature_count=0, diff block is skipped entirely
    
    Examples
    --------
    >>> importance_dict = calculate_residue_rf_importance(models, keep_residues)
    >>> print(f"Residue 100 RF score: {importance_dict[99]['rf_score']:.6f}")
    """
    # Get kept residue positions
    kept_indices = np.where(keep_residues)[0]
    n_kept_residues = len(kept_indices)
    
    # Collect feature importances from all models
    all_feature_importances = []
    for model_dict in models:
        rf_model = model_dict['model']
        all_feature_importances.append(rf_model.feature_importances_)
    
    # Calculate importance for each residue
    residue_importance_dict = {}
    
    for i, pos in enumerate(kept_indices):
        # Raw features block: [i*cp_raw : (i+1)*cp_raw]
        raw_start = i * cp_raw_feature_count
        raw_end = raw_start + cp_raw_feature_count
        
        # Diff features block (only if cp_diff_feature_count > 0)
        if cp_diff_feature_count > 0:
            diff_start = n_kept_residues * cp_raw_feature_count + i * cp_diff_feature_count
            diff_end = diff_start + cp_diff_feature_count
        
        # Collect scores from all models
        residue_scores = []
        for feature_imp in all_feature_importances:
            raw_score = np.sum(feature_imp[raw_start:raw_end])
            if cp_diff_feature_count > 0:
                diff_score = np.sum(feature_imp[diff_start:diff_end])
            else:
                diff_score = 0.0
            residue_scores.append(raw_score + diff_score)
        
        # Average across models
        avg_score = np.mean(residue_scores)
        
        residue_importance_dict[pos] = {
            'position': pos + 1,  # 1-based
            'rf_score': avg_score,
            'model_scores': residue_scores
        }
    
    return residue_importance_dict


def calculate_unbiased_on_target_stats(unique_on_targets_array, sgRNA_to_index):
    """
    Calculate unbiased on-target statistics by removing sgRNA duplication.
    
    Since multiple off-targets can share the same sgRNA (and thus the same 
    on-target structure), this function computes statistics using only 
    unique sgRNAs to avoid bias from repeated measurements.
    
    Parameters
    ----------
    unique_on_targets_array : numpy.ndarray
        Array of shape (n_unique_sgrnas, n_residues, 18) containing on-target
        contact probabilities for each unique sgRNA.
    sgRNA_to_index : dict
        Dictionary mapping sgRNA names to their indices in unique_on_targets_array.
    
    Returns
    -------
    unbiased_stats : dict
        Dictionary mapping residue position (0-based) to statistics:
        - 'cp_on_mean_unbiased': float, mean CP across all unique sgRNAs
        - 'cp_on_std_unbiased': float, standard deviation
        - 'cp_on_max_unbiased': float, maximum CP
        - 'cp_on_median_unbiased': float, median CP
        - 'sgRNA_mean_mean': float, mean of per-sgRNA means
        - 'sgRNA_mean_std': float, std of per-sgRNA means
        - 'n_unique_sgrnas': int, number of unique sgRNAs
    
    Notes
    -----
    The function flattens all 18 CP values per residue per sgRNA to compute
    overall statistics, while also tracking per-sgRNA mean values to assess
    variability across different guide RNAs.
    
    Examples
    --------
    >>> stats = calculate_unbiased_on_target_stats(unique_array, sgrna_map)
    >>> print(f"Residue 100 on-target mean: {stats[99]['cp_on_mean_unbiased']:.4f}")
    """
    n_residues = unique_on_targets_array.shape[1]
    n_unique_sgrnas = len(unique_on_targets_array)
    
    unbiased_stats = {}
    
    for residue_pos in range(n_residues):
        # Get all on-target values for this residue across unique sgRNAs
        all_on_values = unique_on_targets_array[:, residue_pos, :]  # (n_sgrnas, 18)
        
        # Flatten all values for overall statistics
        all_on_flat = all_on_values.flatten()
        
        # Calculate per-sgRNA means
        sgRNA_means = np.mean(all_on_values, axis=1)
        
        unbiased_stats[residue_pos] = {
            'cp_on_mean_unbiased': np.mean(all_on_flat),
            'cp_on_std_unbiased': np.std(all_on_flat),
            'cp_on_max_unbiased': np.max(all_on_flat),
            'cp_on_median_unbiased': np.median(all_on_flat),
            'sgRNA_mean_mean': np.mean(sgRNA_means),
            'sgRNA_mean_std': np.std(sgRNA_means),
            'n_unique_sgrnas': n_unique_sgrnas
        }
    
    return unbiased_stats


# =========================== Single Residue Statistics ===========================
def process_single_residue_stats_enhanced(residue_idx, residue_pos, 
                               all_cp_off_data, all_cp_on_data, all_y_labels, all_sgRNA_indices,
                               unique_on_targets_array, keep_residues,
                               region_names, diff_threshold=0.01, query_region_list=None, top_n=3,
                               class1_baseline_array=None, use_class1_baseline=False,
                               cp_raw_top_num=3):
    """
    Calculate comprehensive statistics for a single residue across all samples.
    
    This function computes contact probability statistics, classification-based
    metrics, and variation measures for one residue, considering both overall
    and per-class patterns.
    
    Parameters
    ----------
    residue_idx : int
        Index of residue in kept_residues array (0-based).
    residue_pos : int
        Actual position of residue in full protein sequence (0-based).
    all_cp_off_data : numpy.ndarray
        Off-target CP data, shape (n_samples, n_residues, 18).
    all_cp_on_data : numpy.ndarray
        On-target CP data, shape (n_samples, n_residues, 18).
    all_y_labels : numpy.ndarray
        Classification labels, shape (n_samples,), values in [0, 1, 2, ...].
    all_sgRNA_indices : numpy.ndarray
        sgRNA indices for each sample, shape (n_samples,).
    unique_on_targets_array : numpy.ndarray
        Unique on-target structures, shape (n_unique_sgrnas, n_residues, 18).
    keep_residues : numpy.ndarray
        Boolean array of kept residues.
    region_names : list of str
        Names of 6 nucleic acid regions.
    diff_threshold : float, default=0.01
        Threshold for determining contact enhancement (off - on > threshold).
    query_region_list : list of str, optional
        Subset of regions to analyze. If None, uses all 18 values.
    top_n : int, default=3
        Number of top contact values to average.
    class1_baseline_array : numpy.ndarray, optional
        Per-sgRNA Class_1 mean baseline, shape (n_unique_sgrnas, n_residues, n_cp_dims).
        Used when use_class1_baseline=True (i.e., has_ref=False mode).
    use_class1_baseline : bool, default=False
        If True, uses class1_baseline_array instead of unique_on_targets_array
        as the reference for CP diff calculation.
    cp_raw_top_num : int, default=3
        Number of top-N values stored per region in cp_raw data.
        Determines how many values each region occupies (e.g. 3 for Cas9/A3A/TadA8e).
    
    Returns
    -------
    residue_stat : dict
        Comprehensive statistics dictionary containing:
        - 'residue_idx': Index in kept array
        - 'position': 1-based position
        - 'overall_stats': Overall statistics across all samples
        - 'region_stats': Per-region breakdown
        - 'class_stats': Per-class statistics (Class_1, Class_2, Class_3, ...)
        - 'variation_stats': Within-sample variation metrics
        - 'raw_data': Raw values for downstream analysis
    
    Notes
    -----
    Overall Statistics Include:
    - Mean/std/max of on-target and off-target CP
    - Mean/std/max of CP difference (off - on)
    - Above/below ratios (fraction exceeding threshold)
    - Percentile values (p90, p95)
    
    Class Statistics:
    - Separate statistics for each off-target class
    - Enables detection of class-specific enhancement patterns
    
    Variation Statistics:
    - Within-sample std/CV/range across 18 dimensions
    - Measures consistency of contact across different regions
    
    Examples
    --------
    >>> stats = process_single_residue_stats_enhanced(
    ...     0, 99, cp_off, cp_on, labels, sgrna_idx, unique_on,
    ...     keep_res, regions, diff_threshold=0.01
    ... )
    >>> print(f"Above ratio: {stats['overall_stats']['above_ratio']:.3f}")
    """
    n_samples = len(all_y_labels)
    
    # Region to index mapping
    region_to_index = {
        'sgRNA_dist': 0, 'sgRNA_prox': 1,
        'tsDNA_dist': 2, 'tsDNA_prox': 3,
        'ntsDNA_dist': 4, 'ntsDNA_prox': 5
    }
    
    # Initialize storage
    residue_cp_on_all = []
    residue_cp_off_all = []
    residue_cp_diff_all = []
    residue_above_flags = []
    residue_cp_off_raw_all = []  # For variation statistics
    
    # Determine number of classes
    unique_labels = np.unique(all_y_labels)
    n_classes = len(unique_labels)
    
    # Per-class storage
    residue_stats_by_class = {}
    for class_idx in unique_labels:
        residue_stats_by_class[class_idx] = {
            'cp_on': [], 'cp_off': [], 'cp_diff': [], 'above_flags': []
        }
    
    # Per-region storage
    region_cp_on = {region: [] for region in region_names}
    region_cp_off = {region: [] for region in region_names}
    region_cp_diff = {region: [] for region in region_names}
    region_above_flags = {region: [] for region in region_names}
    
    # Process each sample
    for sample_idx in range(n_samples):
        sgRNA_idx = all_sgRNA_indices[sample_idx]
        y_label = all_y_labels[sample_idx]
        
        # Get on-target values: use class1 baseline if has_ref=False
        if use_class1_baseline:
            on_values = class1_baseline_array[sgRNA_idx, residue_pos, :]  # (18,)
        else:
            on_values = unique_on_targets_array[sgRNA_idx, residue_pos, :]  # (18,)
        off_values = all_cp_off_data[sample_idx, residue_pos, :]  # (18,)
        
        # Store raw off values for variation calculation
        residue_cp_off_raw_all.append(off_values)
        
        # === Overall statistics (based on query_region_list and top_n) ===
        if query_region_list is None:
            selected_on_values = on_values
            selected_off_values = off_values
        else:
            selected_on_values = []
            selected_off_values = []
            for region in query_region_list:
                region_idx = region_to_index[region]
                start_idx = region_idx * 3
                end_idx = start_idx + 3
                selected_on_values.extend(on_values[start_idx:end_idx])
                selected_off_values.extend(off_values[start_idx:end_idx])
            selected_on_values = np.array(selected_on_values)
            selected_off_values = np.array(selected_off_values)
        
        # Top-N strategy
        effective_top_n = min(top_n, len(selected_on_values))
        
        if effective_top_n > 0:
            on_top_indices = np.argpartition(selected_on_values, -effective_top_n)[-effective_top_n:]
            off_top_indices = np.argpartition(selected_off_values, -effective_top_n)[-effective_top_n:]
            
            on_top_values = np.sort(selected_on_values[on_top_indices])[::-1]
            off_top_values = np.sort(selected_off_values[off_top_indices])[::-1]
            
            cp_diffs = off_top_values - on_top_values
            is_above = np.max(cp_diffs) > diff_threshold
            
            on_value = np.mean(on_top_values)
            off_value = np.mean(off_top_values)
            diff_value = np.mean(cp_diffs)
        else:
            on_value = 0
            off_value = 0
            diff_value = 0
            is_above = False
        
        # Store overall statistics
        residue_cp_on_all.append(on_value)
        residue_cp_off_all.append(off_value)
        residue_cp_diff_all.append(diff_value)
        residue_above_flags.append(is_above)
        
        # Store by class
        residue_stats_by_class[y_label]['cp_on'].append(on_value)
        residue_stats_by_class[y_label]['cp_off'].append(off_value)
        residue_stats_by_class[y_label]['cp_diff'].append(diff_value)
        residue_stats_by_class[y_label]['above_flags'].append(is_above)
        
        # === Per-region statistics ===
        for region_idx, region in enumerate(region_names):
            start_idx = region_idx * cp_raw_top_num
            end_idx = start_idx + cp_raw_top_num
            
            region_on_vals = on_values[start_idx:end_idx]
            region_off_vals = off_values[start_idx:end_idx]
            
            # Guard: skip if slice is empty (data has fewer dims than expected)
            if len(region_on_vals) == 0 or len(region_off_vals) == 0:
                region_cp_on[region].append(0)
                region_cp_off[region].append(0)
                region_cp_diff[region].append(0)
                region_above_flags[region].append(False)
                continue
            
            region_top_n = min(top_n, cp_raw_top_num)
            
            if region_top_n > 0:
                on_region_top_indices = np.argpartition(region_on_vals, -region_top_n)[-region_top_n:]
                off_region_top_indices = np.argpartition(region_off_vals, -region_top_n)[-region_top_n:]
                
                on_region_top = np.sort(region_on_vals[on_region_top_indices])[::-1]
                off_region_top = np.sort(region_off_vals[off_region_top_indices])[::-1]
                
                region_diffs = off_region_top - on_region_top
                region_is_above = np.max(region_diffs) > diff_threshold
                
                region_on_value = np.mean(on_region_top)
                region_off_value = np.mean(off_region_top)
                region_diff_value = np.mean(region_diffs)
            else:
                region_on_value = 0
                region_off_value = 0
                region_diff_value = 0
                region_is_above = False
            
            region_cp_on[region].append(region_on_value)
            region_cp_off[region].append(region_off_value)
            region_cp_diff[region].append(region_diff_value)
            region_above_flags[region].append(region_is_above)
    
    # Convert to numpy arrays
    residue_cp_on_all = np.array(residue_cp_on_all)
    residue_cp_off_all = np.array(residue_cp_off_all)
    residue_cp_diff_all = np.array(residue_cp_diff_all)
    residue_above_flags = np.array(residue_above_flags)
    residue_cp_off_raw_all = np.array(residue_cp_off_raw_all)  # (n_samples, 18)
    
    # === Variation statistics ===
    sample_stds = np.std(residue_cp_off_raw_all, axis=1)
    sample_means = np.mean(residue_cp_off_raw_all, axis=1)
    sample_cvs = np.divide(sample_stds, sample_means, 
                          out=np.zeros_like(sample_stds), 
                          where=sample_means!=0)
    sample_ranges = np.max(residue_cp_off_raw_all, axis=1) - np.min(residue_cp_off_raw_all, axis=1)
    
    variation_stats = {
        'std_mean': np.mean(sample_stds),
        'std_std': np.std(sample_stds),
        'std_max': np.max(sample_stds),
        'std_p90': np.percentile(sample_stds, 90),
        'cv_mean': np.mean(sample_cvs),
        'cv_std': np.std(sample_cvs),
        'cv_max': np.max(sample_cvs),
        'range_mean': np.mean(sample_ranges),
        'range_std': np.std(sample_ranges),
        'range_max': np.max(sample_ranges),
    }
    
    # === Overall statistics ===
    above_count = np.sum(residue_above_flags)
    above_ratio = above_count / n_samples
    
    overall_stats = {
        'cp_on_mean': np.mean(residue_cp_on_all),
        'cp_on_std': np.std(residue_cp_on_all),
        'cp_on_max': np.max(residue_cp_on_all) if len(residue_cp_on_all) > 0 else 0,
        'cp_off_mean': np.mean(residue_cp_off_all),
        'cp_off_std': np.std(residue_cp_off_all),
        'cp_off_max': np.max(residue_cp_off_all) if len(residue_cp_off_all) > 0 else 0,
        'cp_diff_mean': np.mean(residue_cp_diff_all),
        'cp_diff_std': np.std(residue_cp_diff_all),
        'cp_diff_max': np.max(residue_cp_diff_all) if len(residue_cp_diff_all) > 0 else 0,
        'cp_diff_p90': np.percentile(residue_cp_diff_all, 90) if len(residue_cp_diff_all) > 0 else 0,
        'cp_diff_p95': np.percentile(residue_cp_diff_all, 95) if len(residue_cp_diff_all) > 0 else 0,
        'above_count': above_count,
        'above_ratio': above_ratio,
        'below_count': np.sum(residue_cp_diff_all < -diff_threshold),
        'below_ratio': np.sum(residue_cp_diff_all < -diff_threshold) / n_samples,
        'neutral_count': n_samples - above_count - np.sum(residue_cp_diff_all < -diff_threshold),
        'positive_ratio': np.sum(residue_cp_diff_all > 0) / n_samples,
        'strong_positive_ratio': np.sum(residue_cp_diff_all > 0.1) / n_samples,
        'query_regions': query_region_list if query_region_list else 'all',
        'top_n': top_n
    }
    
    # Above group detailed statistics
    if above_count > 0:
        above_indices = np.where(residue_above_flags)[0]
        overall_stats['above_diff_mean'] = np.mean(residue_cp_diff_all[above_indices])
        overall_stats['above_diff_std'] = np.std(residue_cp_diff_all[above_indices])
        overall_stats['above_diff_max'] = np.max(residue_cp_diff_all[above_indices])
        overall_stats['above_cp_mean'] = np.mean(residue_cp_off_all[above_indices])
    else:
        overall_stats['above_diff_mean'] = 0
        overall_stats['above_diff_std'] = 0
        overall_stats['above_diff_max'] = 0
        overall_stats['above_cp_mean'] = 0
    
    # === Per-class statistics ===
    class_stats = {}
    
    for class_idx in unique_labels:
        class_data = residue_stats_by_class[class_idx]
        
        if len(class_data['cp_on']) > 0:
            cp_on_class = np.array(class_data['cp_on'])
            cp_off_class = np.array(class_data['cp_off'])
            cp_diff_class = np.array(class_data['cp_diff'])
            above_flags_class = np.array(class_data['above_flags'])
            
            above_count_class = np.sum(above_flags_class)
            class_name = f"Class_{class_idx+1}"  # 1-based display
            
            class_stats[class_name] = {
                'class_value': class_idx,
                'n_samples': len(cp_on_class),
                'cp_on_mean': np.mean(cp_on_class),
                'cp_off_mean': np.mean(cp_off_class),
                'cp_diff_mean': np.mean(cp_diff_class),
                'cp_diff_std': np.std(cp_diff_class),
                'cp_diff_max': np.max(cp_diff_class),
                'above_count': above_count_class,
                'above_ratio': above_count_class / len(cp_on_class),
                'positive_ratio': np.sum(cp_diff_class > 0) / len(cp_diff_class)
            }
        else:
            class_name = f"Class_{class_idx+1}"
            class_stats[class_name] = {
                'class_value': class_idx,
                'n_samples': 0,
                'cp_on_mean': 0,
                'cp_off_mean': 0,
                'cp_diff_mean': 0,
                'cp_diff_std': 0,
                'cp_diff_max': 0,
                'above_count': 0,
                'above_ratio': 0,
                'positive_ratio': 0
            }
    
    # === Per-region statistics ===
    region_stats = {}
    for region in region_names:
        region_on = np.array(region_cp_on[region])
        region_off = np.array(region_cp_off[region])
        region_diff = np.array(region_cp_diff[region])
        region_above = np.array(region_above_flags[region])
        
        region_above_count = np.sum(region_above)
        region_above_ratio = region_above_count / n_samples
        
        region_stats[region] = {
            'cp_on_mean': np.mean(region_on),
            'cp_on_std': np.std(region_on),
            'cp_on_max': np.max(region_on) if len(region_on) > 0 else 0,
            'cp_off_mean': np.mean(region_off),
            'cp_off_std': np.std(region_off),
            'cp_off_max': np.max(region_off) if len(region_off) > 0 else 0,
            'cp_diff_mean': np.mean(region_diff),
            'cp_diff_std': np.std(region_diff),
            'cp_diff_max': np.max(region_diff) if len(region_diff) > 0 else 0,
            'cp_diff_p90': np.percentile(region_diff, 90) if len(region_diff) > 0 else 0,
            'above_count': region_above_count,
            'above_ratio': region_above_ratio,
            'below_ratio': np.sum(region_diff < -diff_threshold) / n_samples,
            'positive_ratio': np.sum(region_diff > 0) / n_samples
        }
        
        if region_above_count > 0:
            region_above_indices = np.where(region_above)[0]
            region_stats[region]['above_diff_mean'] = np.mean(region_diff[region_above_indices])
            region_stats[region]['above_diff_max'] = np.max(region_diff[region_above_indices])
            region_stats[region]['above_cp_mean'] = np.mean(region_off[region_above_indices])
        else:
            region_stats[region]['above_diff_mean'] = 0
            region_stats[region]['above_diff_max'] = 0
            region_stats[region]['above_cp_mean'] = 0
    
    # Assemble final result
    residue_stat = {
        'residue_idx': residue_idx,
        'position': residue_pos + 1,  # 1-based
        'overall_stats': overall_stats,
        'region_stats': region_stats,
        'class_stats': class_stats,
        'variation_stats': variation_stats,
        'diff_threshold': diff_threshold,
        'top_n': top_n,
        'n_classes': n_classes,
        'raw_data': {
            'cp_on_values': residue_cp_on_all,
            'cp_off_values': residue_cp_off_all,
            'cp_diff_values': residue_cp_diff_all,
            'above_flags': residue_above_flags,
            'cp_off_raw_all': residue_cp_off_raw_all
        }
    }
    
    return residue_stat


# =========================== Parallel Processing ===========================
def residue_level_stats_parallel(data_list, keep_residues, models, 
                               cpu_threads=None, protein_name="Cas", 
                               diff_threshold=0.01, query_region_list=None, 
                               top_n=3, target_key='y_g3', verbose=False,
                               has_ref=True,
                               cp_raw_feature_count=18, cp_diff_feature_count=6, cp_raw_top_num=3):
    """
    Calculate comprehensive residue-level statistics in parallel.
    
    This function coordinates parallel processing of all retained residues,
    computing RF importance, unbiased on-target statistics, and detailed
    per-residue contact probability metrics.
    
    Parameters
    ----------
    data_list : list of dict
        List of data dictionaries from multiple seeds. Each dict must contain:
        - target_key: Classification labels
        - 'cp_raw_cas_nuc': Off-target CP arrays
        - 'cp_raw_on_cas_nuc': On-target CP arrays
        - 'key_info': (sgRNA_name, off_target_location) tuples
    keep_residues : numpy.ndarray
        Boolean array indicating retained residues.
    models : list of dict
        Trained Random Forest models for importance calculation.
    cpu_threads : int, optional
        Number of CPU cores to use. If None, uses (total_cores - 2).
    protein_name : str, default="Cas"
        Protein name for display.
    diff_threshold : float, default=0.01
        Threshold for contact enhancement detection.
    query_region_list : list of str, optional
        Subset of regions to analyze. If None, uses all 18 values.
    top_n : int, default=3
        Number of top contact values to average.
    target_key : str, default='y_g3'
        Classification label key ('y_g2', 'y_g3', 'y_g4', etc.).
    verbose : bool, default=False
        If True, prints detailed progress information.
    has_ref : bool, default=True
        If True, uses on-target reference structures (unique_on_targets_array) as
        baseline for CP diff calculation (original behaviour).
        If False, computes per-sgRNA Class_1 mean from off-target data and uses
        that as the baseline. above_ratio definition is unchanged.
    cp_raw_feature_count : int, default=18
        Number of raw CP features per residue (18 for Cas9, 9 for A3A, 3 for TadA8e).
    cp_diff_feature_count : int, default=6
        Number of diff CP features per residue (6 for Cas9, 0 for A3A/TadA8e).
    cp_raw_top_num : int, default=3
        Number of top-N values stored per region in cp_raw data.
        n_regions = cp_raw_feature_count // cp_raw_top_num.
    
    Returns
    -------
    results : dict
        Comprehensive analysis results containing:
        - 'residue_stats': Dict mapping position (0-based) to full statistics
        - 'summary_stats': Overall summary metrics
        - 'protein_name': Protein name
        - 'n_residues': Number of analyzed residues
        - 'diff_threshold': Threshold used
        - 'query_regions': Regions analyzed
        - 'top_n': Top-N parameter
        - 'target_key': Label key used
        - 'sgRNA_to_index': sgRNA name to index mapping
        - 'unique_on_targets_array': Unique on-target structures
    
    Notes
    -----
    Processing Pipeline:
    1. Build sgRNA index and extract unique on-targets
    2. Collect all off-target data across seeds
    3. Calculate RF importance for each residue
    4. Calculate unbiased on-target statistics
    5. Parallel compute per-residue detailed statistics
    6. Compute summary statistics
    
    The function automatically detects the number of classes from the data
    and adapts class-based statistics accordingly.
    
    Examples
    --------
    >>> results = residue_level_stats_parallel(
    ...     data_list, keep_residues, models,
    ...     protein_name="Cas9", verbose=True
    ... )
    >>> print(f"Analyzed {results['n_residues']} residues")
    """
    
    if verbose:
        print(f"\nENHANCED RESIDUE-LEVEL STATISTICS for {protein_name}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting analysis...")
        print(f"Using target key: {target_key}")
    
    # Set CPU threads
    if cpu_threads is None:
        cpu_threads = max(1, multiprocessing.cpu_count() - 2)
    
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using {cpu_threads} CPU cores")
    
    # 1. Preprocess data
    start_time = datetime.now()
    if verbose:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing data...")
    
    # Build sgRNA index and unique on-targets array
    sgRNA_to_index = {}
    unique_on_targets_list = []

    for data_dict in data_list:
        n_samples = len(data_dict[target_key])
        for sample_idx in range(n_samples):
            sgRNA_name, _ = data_dict['key_info'][sample_idx]

            if sgRNA_name not in sgRNA_to_index:
                sgRNA_to_index[sgRNA_name] = len(sgRNA_to_index)
                if has_ref:
                    # Only collect on-target structures when ref is available
                    unique_on_targets_list.append(data_dict['cp_raw_on_cas_nuc'][sample_idx])

    if has_ref:
        unique_on_targets_array = np.array(unique_on_targets_list)
    else:
        # Placeholder; will be replaced by class1_baseline_array later
        unique_on_targets_array = None

    # Collect all off-target data
    all_cp_off_data = []
    all_cp_on_data = []
    all_y_labels = []
    all_sgRNA_indices = []

    for data_dict in data_list:
        n_samples = len(data_dict[target_key])

        for sample_idx in range(n_samples):
            sgRNA_name, _ = data_dict['key_info'][sample_idx]
            all_cp_off_data.append(data_dict['cp_raw_cas_nuc'][sample_idx])
            if has_ref:
                all_cp_on_data.append(data_dict['cp_raw_on_cas_nuc'][sample_idx])
            all_y_labels.append(data_dict[target_key][sample_idx])
            all_sgRNA_indices.append(sgRNA_to_index[sgRNA_name])
    
    # Convert to numpy arrays (labels are 1-based, convert to 0-based)
    all_cp_off_data = np.array(all_cp_off_data)
    all_cp_on_data = np.array(all_cp_on_data) if has_ref else None
    all_y_labels = np.array(all_y_labels) - 1  # Convert to 0-based
    all_sgRNA_indices = np.array(all_sgRNA_indices)
    
    # Define region names dynamically based on cp_raw_feature_count and cp_raw_top_num
    _all_region_names = ['sgRNA_dist', 'sgRNA_prox', 'tsDNA_dist', 'tsDNA_prox', 'ntsDNA_dist', 'ntsDNA_prox']
    n_regions = cp_raw_feature_count // cp_raw_top_num
    if n_regions <= len(_all_region_names):
        region_names = _all_region_names[:n_regions]
    else:
        region_names = _all_region_names + [f'region_{i}' for i in range(len(_all_region_names), n_regions)]
    
    # Get kept residues
    kept_indices = np.where(keep_residues)[0]
    
    # Detect number of classes
    unique_classes = np.unique(all_y_labels)
    n_classes = len(unique_classes)
    
    if verbose:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing completed")
        print(f"  Protein length: {len(keep_residues)} amino acids")
        print(f"  Kept residues: {len(kept_indices)}")
        print(f"  Total samples: {len(all_y_labels)}")
        print(f"  Unique sgRNAs: {len(sgRNA_to_index)}")
        print(f"  Number of classes: {n_classes}")
        print(f"  has_ref: {has_ref}")
        
        # Print class distribution
        class_dist = np.bincount(all_y_labels)
        print(f"  Class distribution:")
        for i, count in enumerate(class_dist):
            print(f"    Class_{i+1}: {count} samples")
    
    # 2. Calculate RF importance
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating RF importance...")
    residue_rf_importance = calculate_residue_rf_importance(
        models, keep_residues,
        cp_raw_feature_count=cp_raw_feature_count,
        cp_diff_feature_count=cp_diff_feature_count
    )
    
    # 3. Calculate unbiased on-target statistics (only when ref is available)
    if has_ref:
        if verbose:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating unbiased on-target statistics...")
        unbiased_on_stats = calculate_unbiased_on_target_stats(unique_on_targets_array, sgRNA_to_index)
    else:
        unbiased_on_stats = {}
        if verbose:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] has_ref=False: skipping unbiased on-target statistics")
    
    # 3b. If no ref, compute per-sgRNA Class_1 baseline from off-target data
    class1_baseline_array = None
    use_class1_baseline = False
    
    if not has_ref:
        use_class1_baseline = True
        if verbose:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] has_ref=False: computing per-sgRNA Class_1 baseline...")
        
        n_unique_sgrnas = len(sgRNA_to_index)
        n_residues = all_cp_off_data.shape[1]
        n_cp_dims = all_cp_off_data.shape[2]  # infer from data (e.g. 18 for Cas9, 9 for A3A)
        
        # Global Class_1 mean as fallback
        global_class1_mask = (all_y_labels == 0)
        if global_class1_mask.sum() > 0:
            global_class1_mean = np.mean(all_cp_off_data[global_class1_mask], axis=0)  # (n_residues, n_cp_dims)
        else:
            global_class1_mean = np.zeros((n_residues, n_cp_dims))
        
        class1_baseline_array = np.zeros((n_unique_sgrnas, n_residues, n_cp_dims))
        
        fallback_count = 0
        for sgRNA_name, sgRNA_idx in sgRNA_to_index.items():
            mask = (all_sgRNA_indices == sgRNA_idx) & (all_y_labels == 0)
            if mask.sum() > 0:
                class1_baseline_array[sgRNA_idx] = np.mean(all_cp_off_data[mask], axis=0)
            else:
                # Fallback: use global Class_1 mean
                class1_baseline_array[sgRNA_idx] = global_class1_mean
                fallback_count += 1
        
        if verbose:
            print(f"  Per-sgRNA Class_1 baseline computed for {n_unique_sgrnas - fallback_count}/{n_unique_sgrnas} sgRNAs")
            if fallback_count > 0:
                print(f"  WARNING: {fallback_count} sgRNAs had no Class_1 samples, used global Class_1 mean as fallback")
    
    # 4. Parallel calculate residue statistics
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting parallel residue statistics...")
    
    # Prepare parallel tasks
    residue_data = []
    for idx, pos in enumerate(kept_indices):
        residue_data.append((idx, pos))
    
    # Parallel processing
    parallel_start_time = datetime.now()
    
    residue_stats_list = Parallel(n_jobs=cpu_threads, backend='loky', verbose=1 if verbose else 0)(
        delayed(process_single_residue_stats_enhanced)(
            residue_idx, residue_pos,
            all_cp_off_data, all_cp_on_data, all_y_labels, all_sgRNA_indices,
            unique_on_targets_array, keep_residues,
            region_names, diff_threshold, query_region_list, top_n,
            class1_baseline_array, use_class1_baseline,
            cp_raw_top_num
        ) for residue_idx, residue_pos in residue_data
    )
    
    if verbose:
        parallel_time = (datetime.now() - parallel_start_time).total_seconds()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Parallel processing completed in {parallel_time:.2f} seconds")
    
    # 5. Integrate results
    residue_stats_dict = {}
    
    for residue_stat in residue_stats_list:
        pos = residue_stat['position'] - 1  # Convert back to 0-based
        
        # Add RF importance
        if pos in residue_rf_importance:
            residue_stat['rf_importance'] = residue_rf_importance[pos]
        
        # Add unbiased on-target statistics
        if pos in unbiased_on_stats:
            residue_stat['unbiased_on_stats'] = unbiased_on_stats[pos]
        
        residue_stats_dict[pos] = residue_stat
    
    # 6. Calculate summary statistics
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing summary statistics...")
    
    all_above_ratios = [r['overall_stats']['above_ratio'] for r in residue_stats_dict.values()]
    all_rf_scores = [r['rf_importance']['rf_score'] for r in residue_stats_dict.values() if 'rf_importance' in r]
    all_cv_means = [r['variation_stats']['cv_mean'] for r in residue_stats_dict.values()]
    
    summary_stats = {
        'n_residues': len(residue_stats_dict),
        'mean_above_ratio': np.mean(all_above_ratios),
        'std_above_ratio': np.std(all_above_ratios),
        'mean_rf_score': np.mean(all_rf_scores),
        'std_rf_score': np.std(all_rf_scores),
        'mean_cv': np.mean(all_cv_means),
        'std_cv': np.std(all_cv_means),
        'high_above_ratio_residues': sum(1 for r in all_above_ratios if r > 0.5),
        'high_rf_residues': sum(1 for s in all_rf_scores if s > np.mean(all_rf_scores) + np.std(all_rf_scores)),
        'high_variation_residues': sum(1 for cv in all_cv_means if cv > np.mean(all_cv_means) + np.std(all_cv_means)),
        'query_regions': query_region_list if query_region_list else 'all',
        'top_n': top_n,
        'target_key': target_key,
        'n_classes': n_classes
    }
    
    if verbose:
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total analysis completed in {total_time:.2f} seconds")
        print(f"Performance: {len(kept_indices)/total_time:.1f} residues/second")
    
    # Return results
    return {
        'residue_stats': residue_stats_dict,
        'summary_stats': summary_stats,
        'protein_name': protein_name,
        'n_residues': len(residue_stats_dict),
        'diff_threshold': diff_threshold,
        'query_regions': query_region_list,
        'top_n': top_n,
        'target_key': target_key,
        'sgRNA_to_index': sgRNA_to_index,
        'unique_on_targets_array': unique_on_targets_array
    }


# =========================== Contact Enhancement Analysis ===========================
def residue_contact_enhancement(data_list, keep_residues, models, protein_sequence,
                                ccr_ranking_df, cpu_threads=None, protein_name="Cas",
                                diff_threshold=0.01, query_region_list=None, top_n=3,
                                target_key='y_g3', bottom_percentile=10, verbose=False,
                                has_ref=True,
                                cp_raw_feature_count=18, cp_diff_feature_count=6, cp_raw_top_num=3):
    """
    Calculate residue-level contact enhancement metric and create comprehensive DataFrame.
    
    This function integrates the complete pipeline:
    1. Calculates residue-level statistics using residue_level_stats_parallel
    2. Computes relative RF importance (log2-scaled)
    3. Calculates contact enhancement metric (max - min of class above_ratios)
    4. Maps residues to consensus contact regions (CCRs)
    5. Returns comprehensive analysis DataFrame
    
    Parameters
    ----------
    data_list : list of dict
        List of data dictionaries from multiple seeds. Each dict must contain:
        - target_key: Classification labels
        - 'cp_raw_cas_nuc': Off-target CP arrays
        - 'cp_raw_on_cas_nuc': On-target CP arrays
        - 'key_info': (sgRNA_name, off_target_location) tuples
    keep_residues : numpy.ndarray
        Boolean array indicating retained residues.
    models : list of dict
        Trained Random Forest models for importance calculation.
    protein_sequence : str
        Complete protein amino acid sequence (single-letter codes).
    ccr_ranking_df : pandas.DataFrame
        Output from ranking_consensus_contact_region, containing CCR information
        with columns: 'CCR_ID', 'CCR_range', 'CCR_all_residues', etc.
    cpu_threads : int, optional
        Number of CPU cores to use. If None, uses (total_cores - 2).
    protein_name : str, default="Cas"
        Protein name for display and logging.
    diff_threshold : float, default=0.01
        Threshold for contact enhancement detection.
    query_region_list : list of str, optional
        Subset of regions to analyze. If None, uses all 18 values.
    top_n : int, default=3
        Number of top contact values to average.
    target_key : str, default='y_g3'
        Classification label key ('y_g2', 'y_g3', 'y_g4', etc.).
    bottom_percentile : float, default=10
        Percentile (0-100) of lowest importance scores to use as baseline
        for relative importance calculation. Default 10 means bottom 10%.
    verbose : bool, default=False
        If True, prints detailed analysis progress and summary statistics.
    has_ref : bool, default=True
        If True, uses on-target reference structures as baseline for CP diff
        calculation (original behaviour, requires cp_raw_on_cas_nuc).
        If False, computes per-sgRNA Class_1 mean from off-target data as
        the reference baseline. above_ratio definition is unchanged.
    cp_raw_feature_count : int, default=18
        Number of raw CP features per residue (18 for Cas9, 9 for A3A, 3 for TadA8e).
    cp_diff_feature_count : int, default=6
        Number of diff CP features per residue (6 for Cas9, 0 for A3A/TadA8e).
    cp_raw_top_num : int, default=3
        Number of top-N values stored per region in cp_raw data.
        n_regions = cp_raw_feature_count // cp_raw_top_num.
    
    Returns
    -------
    df : pandas.DataFrame
        Comprehensive residue analysis table with columns:
        - 'residue_index': int, 1-based residue position
        - 'residue_name': str, format like "1020K" (position + amino acid)
        - 'CCR_index': int, 1-based CCR identifier (0 if not in any CCR)
        - 'CCR_range': str, CCR position range (e.g., "1020-1025")
        - 'relative_importance': float, log2(importance/baseline), 4 decimals
        - 'contact_enhance': float, effective contact enhancement (0-100%, 2 decimals)
    
    Notes
    -----
    Relative Importance Calculation:
    1. Calculate raw RF importance for all residues
    2. Find bottom X% (default 10%) of importance scores
    3. Compute baseline = mean(bottom X% scores)
    4. Relative importance = log2(raw_importance / baseline)
    5. This normalization:
       - Centers low-importance residues around 0
       - Amplifies differences for high-importance residues
       - Makes scores comparable across different proteins
    
    Contact Enhancement Calculation:
    - For each residue, extracts above_ratio from all classes
    - Computes: max(above_ratios) - min(above_ratios)
    - Represents the maximum spread in enhancement across classes
    - High values indicate class-specific contact enhancement
    
    Biological Interpretation:
    - High relative importance → Strongly affects off-target prediction
    - High contact enhancement → Differential behavior across off-target strengths
    - Both high → Strong candidate for mutagenesis to reduce off-targets
    
    CCR Assignment:
    - Parses CCR_all_residues column to extract residue positions
    - Assigns each residue to its corresponding CCR
    - Residues not in any CCR are marked with CCR_index=0
    
    Examples
    --------
    >>> # Basic usage
    >>> df = residue_contact_enhancement(
    ...     data_list, keep_residues, models, cas9_sequence, ccr_df,
    ...     protein_name="Cas9", verbose=True
    ... )
    >>> print(df.head())
    
    >>> # Custom baseline percentile
    >>> df = residue_contact_enhancement(
    ...     data_list, keep_residues, models, cas9_sequence, ccr_df,
    ...     protein_name="Cas9", bottom_percentile=5, verbose=False
    ... )
    
    >>> # Find top candidates
    >>> top_candidates = df.nlargest(20, 'contact_enhance')
    >>> high_importance = df[df['relative_importance'] > 2.0]
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESIDUE CONTACT ENHANCEMENT ANALYSIS")
        print(f"{'='*70}")
        print(f"Protein: {protein_name}")
        print(f"Bottom percentile for baseline: {bottom_percentile}%")
    
    # Step 1: Calculate residue-level statistics
    if verbose:
        print(f"\nStep 1: Calculating residue-level statistics...")
    
    residue_stats_results = residue_level_stats_parallel(
        data_list=data_list,
        keep_residues=keep_residues,
        models=models,
        cpu_threads=cpu_threads,
        protein_name=protein_name,
        diff_threshold=diff_threshold,
        query_region_list=query_region_list,
        top_n=top_n,
        target_key=target_key,
        verbose=verbose,
        has_ref=has_ref,
        cp_raw_feature_count=cp_raw_feature_count,
        cp_diff_feature_count=cp_diff_feature_count,
        cp_raw_top_num=cp_raw_top_num
    )
    
    residue_stats = residue_stats_results['residue_stats']
    
    if verbose:
        print(f"✓ Analyzed {len(residue_stats)} residues")
    
    # Step 2: Calculate relative importance
    if verbose:
        print(f"\nStep 2: Calculating relative importance...")
    
    # Collect all raw importance scores
    all_importance_scores = []
    importance_map = {}
    
    for pos_0based, stats in residue_stats.items():
        rf_score = stats.get('rf_importance', {}).get('rf_score', 0.0)
        all_importance_scores.append(rf_score)
        importance_map[pos_0based] = rf_score
    
    all_importance_scores = np.array(all_importance_scores)
    
    # Calculate baseline from bottom percentile
    percentile_threshold = np.percentile(all_importance_scores, bottom_percentile)
    bottom_scores = all_importance_scores[all_importance_scores <= percentile_threshold]
    
    if len(bottom_scores) == 0:
        # Fallback: use minimum if no scores in bottom percentile
        baseline_importance = np.min(all_importance_scores)
    else:
        baseline_importance = np.mean(bottom_scores)
    
    # Prevent division by zero
    if baseline_importance == 0:
        baseline_importance = 1e-10
    
    if verbose:
        print(f"  Raw importance range: [{np.min(all_importance_scores):.6f}, "
              f"{np.max(all_importance_scores):.6f}]")
        print(f"  Bottom {bottom_percentile}% threshold: {percentile_threshold:.6f}")
        print(f"  Baseline (mean of bottom {bottom_percentile}%): {baseline_importance:.6f}")
        print(f"  Number of residues in baseline: {len(bottom_scores)}")
    
    # Calculate relative importance for all residues
    relative_importance_map = {}
    for pos_0based, raw_importance in importance_map.items():
        # Prevent log2(0)
        if raw_importance <= 0:
            raw_importance = 1e-10
        relative_imp = np.log2(raw_importance / baseline_importance)
        relative_importance_map[pos_0based] = relative_imp
    
    if verbose:
        all_rel_imp = np.array(list(relative_importance_map.values()))
        print(f"  Relative importance range: [{np.min(all_rel_imp):.4f}, "
              f"{np.max(all_rel_imp):.4f}]")
    
    # Step 3: Create position to CCR mapping
    if verbose:
        print(f"\nStep 3: Mapping residues to CCRs...")
    
    position_to_ccr = {}
    
    for _, row in ccr_ranking_df.iterrows():
        ccr_id = row['CCR_ID']
        ccr_range = row['CCR_range']
        ccr_all_residues = row['CCR_all_residues']
        
        # Parse residue positions from CCR_all_residues
        # Format: "1020K,1021T,1022G,..."
        if pd.notna(ccr_all_residues) and ccr_all_residues != "":
            residue_items = ccr_all_residues.split(',')
            for item in residue_items:
                item = item.strip()
                if item:
                    # Extract position (remove amino acid letter)
                    pos_str = ''.join(filter(str.isdigit, item))
                    if pos_str:
                        pos_1based = int(pos_str)
                        position_to_ccr[pos_1based] = {
                            'CCR_index': ccr_id,
                            'CCR_range': ccr_range
                        }
    
    if verbose:
        print(f"  Mapped {len(position_to_ccr)} residues to {len(ccr_ranking_df)} CCRs")
    
    # Step 4: Build comprehensive DataFrame
    if verbose:
        print(f"\nStep 4: Building comprehensive DataFrame...")
    
    data_rows = []
    
    for pos_0based, stats in residue_stats.items():
        position_1based = stats['position']
        
        # Get amino acid
        if position_1based <= len(protein_sequence):
            amino_acid = protein_sequence[position_1based - 1]
            residue_name = f"{position_1based}{amino_acid}"
        else:
            residue_name = f"{position_1based}X"
        
        # Get CCR information
        if position_1based in position_to_ccr:
            ccr_info = position_to_ccr[position_1based]
            ccr_index = ccr_info['CCR_index']
            ccr_range = ccr_info['CCR_range']
        else:
            ccr_index = 0
            ccr_range = "N/A"
        
        # Get relative importance
        rel_importance = relative_importance_map.get(pos_0based, 0.0)
        
        # Calculate contact enhancement (exploration_3)
        class_stats = stats.get('class_stats', {})
        class_above_ratios = []
        
        # Extract above_ratios from all classes
        for class_name in sorted(class_stats.keys()):
            above_ratio = class_stats[class_name]['above_ratio']
            class_above_ratios.append(above_ratio)
        
        # Calculate enhancement metric
        if len(class_above_ratios) >= 2:
            contact_enhance = max(class_above_ratios) - min(class_above_ratios)
        else:
            contact_enhance = 0.0
        
        # Convert to percentage (0-100%)
        contact_enhance_pct = contact_enhance * 100.0
        
        # Create row
        data_rows.append({
            'residue_index': position_1based,
            'residue_name': residue_name,
            'CCR_index': ccr_index,
            'CCR_range': ccr_range,
            'relative_importance': round(rel_importance, 4),
            'contact_enhance': round(contact_enhance_pct, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    df = df.sort_values('residue_index')
    
    # Step 5: Print summary statistics
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Total residues: {len(df)}")
        print(f"Residues in CCRs: {(df['CCR_index'] > 0).sum()}")
        print(f"Residues not in CCRs: {(df['CCR_index'] == 0).sum()}")
        
        print(f"\nRelative importance statistics:")
        print(f"  Mean: {df['relative_importance'].mean():.4f}")
        print(f"  Std: {df['relative_importance'].std():.4f}")
        print(f"  Min: {df['relative_importance'].min():.4f}")
        print(f"  Max: {df['relative_importance'].max():.4f}")
        print(f"  Median: {df['relative_importance'].median():.4f}")
        print(f"  Residues with rel_imp > 0: {(df['relative_importance'] > 0).sum()}")
        print(f"  Residues with rel_imp > 2: {(df['relative_importance'] > 2).sum()}")
        
        print(f"\nContact enhancement statistics:")
        print(f"  Mean: {df['contact_enhance'].mean():.2f}%")
        print(f"  Std: {df['contact_enhance'].std():.2f}%")
        print(f"  Max: {df['contact_enhance'].max():.2f}%")
        print(f"  Residues with >30% enhancement: {(df['contact_enhance'] > 30).sum()}")
        print(f"  Residues with >50% enhancement: {(df['contact_enhance'] > 50).sum()}")
        print(f"{'='*70}")
            
    return df

# =========================== Residue CP Stats by Class ===========================
def _safe_logit(p, eps=1e-6):
    """
    Compute natural log logit with boundary clipping to avoid inf/-inf.

    logit(p) = ln(p / (1 - p))

    Parameters
    ----------
    p : float or numpy scalar
        Probability value in [0, 1].
    eps : float, default=1e-6
        Clipping boundary to avoid log(0).

    Returns
    -------
    float
        logit(p) clipped at boundaries.
    """
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return float(np.log(p_clipped / (1.0 - p_clipped)))
def _compute_residue_cp_stats(residue_pos, all_cp_off_data, all_y_labels,
                               unique_labels, unique_on_targets_array,
                               all_sgRNA_indices, has_ref, count_n_top, cp_raw_top_num_val):
    """
    Compute per-class raw CP statistics for a single residue.

    Parameters
    ----------
    residue_pos : int
        0-based position of the residue in the protein sequence.
    all_cp_off_data : numpy.ndarray
        Off-target CP data, shape (n_samples, n_residues, n_cp_dims).
    all_y_labels : numpy.ndarray
        0-based class labels, shape (n_samples,).
    unique_labels : numpy.ndarray
        Sorted unique class labels (0-based).
    unique_on_targets_array : numpy.ndarray or None
        Unique on-target CP data, shape (n_unique_sgrnas, n_residues, n_cp_dims).
        None when has_ref=False.
    all_sgRNA_indices : numpy.ndarray
        sgRNA index for each sample, shape (n_samples,).
    has_ref : bool
        Whether on-target reference data is available.
    count_n_top : int
        Number of top samples to average for _n columns.
    cp_raw_top_num_val : int
        Number of top values to sum for cp_top3_sum (the "top-N sum" parameter).

    Returns
    -------
    result : dict
        Dictionary with all CP statistics for this residue.
    """
    result = {'residue_pos': residue_pos}

    # Extract raw CP vector for this residue across all samples: (n_samples, n_cp_dims)
    raw_vecs = all_cp_off_data[:, residue_pos, :]
    n_samples = len(all_y_labels)

    # Compute per-sample scalar summaries
    cp_max_vals = np.max(raw_vecs, axis=1)  # (n_samples,)

    # top-N sum: global top cp_raw_top_num_val values across all dims
    if raw_vecs.shape[1] >= cp_raw_top_num_val:
        # argpartition is faster than full sort
        top_indices = np.argpartition(raw_vecs, -cp_raw_top_num_val, axis=1)[:, -cp_raw_top_num_val:]
        cp_topn_sum_vals = np.sum(np.take_along_axis(raw_vecs, top_indices, axis=1), axis=1)
    else:
        cp_topn_sum_vals = np.sum(raw_vecs, axis=1)

    # Per-class statistics
    for class_idx in unique_labels:
        class_name = f"class{class_idx + 1}"  # 1-based display
        mask = (all_y_labels == class_idx)
        class_max = cp_max_vals[mask]
        class_topn = cp_topn_sum_vals[mask]

        if len(class_max) == 0:
            result[f"{class_name}_cp_max"] = np.nan
            result[f"{class_name}_cp_top3_sum"] = np.nan
            result[f"{class_name}_cp_max_n"] = np.nan
            result[f"{class_name}_cp_top3_sum_n"] = np.nan
            continue

        # Mean over all samples in this class
        result[f"{class_name}_cp_max"] = float(np.mean(class_max))
        result[f"{class_name}_cp_top3_sum"] = float(np.mean(class_topn))

        # Top count_n_top samples (sorted descending, take first count_n_top)
        n_take = min(count_n_top, len(class_max))

        top_max_indices = np.argpartition(class_max, -n_take)[-n_take:]
        result[f"{class_name}_cp_max_n"] = float(np.mean(class_max[top_max_indices]))

        top_topn_indices = np.argpartition(class_topn, -n_take)[-n_take:]
        result[f"{class_name}_cp_top3_sum_n"] = float(np.mean(class_topn[top_topn_indices]))

    # Ref statistics (has_ref=True only, deduplicated by sgRNA)
    if has_ref and unique_on_targets_array is not None:
        # unique_on_targets_array is already deduplicated (one row per unique sgRNA)
        ref_vecs = unique_on_targets_array[:, residue_pos, :]  # (n_unique_sgrnas, n_cp_dims)
        n_unique = ref_vecs.shape[0]

        ref_max_vals = np.max(ref_vecs, axis=1)  # (n_unique_sgrnas,)

        if ref_vecs.shape[1] >= cp_raw_top_num_val:
            top_idx = np.argpartition(ref_vecs, -cp_raw_top_num_val, axis=1)[:, -cp_raw_top_num_val:]
            ref_topn_vals = np.sum(np.take_along_axis(ref_vecs, top_idx, axis=1), axis=1)
        else:
            ref_topn_vals = np.sum(ref_vecs, axis=1)

        result['ref_cp_max'] = float(np.mean(ref_max_vals))
        result['ref_cp_top3_sum'] = float(np.mean(ref_topn_vals))

        n_take_ref = min(count_n_top, n_unique)
        top_ref_max_idx = np.argpartition(ref_max_vals, -n_take_ref)[-n_take_ref:]
        result['ref_cp_max_n'] = float(np.mean(ref_max_vals[top_ref_max_idx]))

        top_ref_topn_idx = np.argpartition(ref_topn_vals, -n_take_ref)[-n_take_ref:]
        result['ref_cp_top3_sum_n'] = float(np.mean(ref_topn_vals[top_ref_topn_idx]))

    return result


def stats_residue_cp_by_type(data_list, keep_residues, models, protein_sequence,
                              ccr_ranking_df, cpu_threads=None, protein_name="Cas",
                              diff_threshold=0.01, query_region_list=None, top_n=3,
                              target_key='y_g3', bottom_percentile=10, verbose=False,
                              has_ref=True,
                              cp_raw_feature_count=18, cp_diff_feature_count=6, cp_raw_top_num=3,
                              count_n_top=50):
    """
    Compute residue-level raw CP statistics broken down by class (and ref).

    For each kept residue, calculates per-class summary statistics of the raw
    contact probability values, along with RF relative importance, CCR mapping,
    and contact_enhance (max - min of class above_ratios).

    Parameters
    ----------
    data_list : list of dict
        List of data dictionaries from multiple seeds. Each dict must contain:
        - target_key: Classification labels
        - 'cp_raw_cas_nuc': Off-target CP arrays
        - 'cp_raw_on_cas_nuc': On-target CP arrays (required when has_ref=True)
        - 'key_info': (sgRNA_name, off_target_location) tuples
    keep_residues : numpy.ndarray
        Boolean array indicating retained residues.
    models : list of dict
        Trained Random Forest models for importance calculation.
    protein_sequence : str
        Complete protein amino acid sequence (single-letter codes).
    ccr_ranking_df : pandas.DataFrame
        CCR information with columns: 'CCR_ID', 'CCR_range', 'CCR_all_residues'.
    cpu_threads : int, optional
        Number of CPU cores for parallel processing. Default: total_cores - 2.
    protein_name : str, default="Cas"
        Protein name for display and logging.
    diff_threshold : float, default=0.01
        Threshold for contact enhancement detection (above_ratio calculation).
    query_region_list : list of str, optional
        Subset of regions for above_ratio calculation. None uses all regions.
    top_n : int, default=3
        Top-N parameter for above_ratio calculation.
    target_key : str, default='y_g3'
        Classification label key.
    bottom_percentile : float, default=10
        Bottom percentile of RF scores used as baseline for relative_importance.
    verbose : bool, default=False
        If True, print detailed progress.
    has_ref : bool, default=True
        If True, uses on-target reference structures and outputs ref_* columns.
        If False, ref_* columns are not output.
    cp_raw_feature_count : int, default=18
        Number of raw CP features per residue (18 for Cas9, 9 for A3A, 3 for TadA8e).
    cp_diff_feature_count : int, default=6
        Number of diff CP features per residue (6 for Cas9, 0 for A3A/TadA8e).
    cp_raw_top_num : int, default=3
        Number of top values per region; also used as top-N for cp_top3_sum.
        n_regions = cp_raw_feature_count // cp_raw_top_num.
    count_n_top : int, default=50
        Number of top-ranked samples to include in _n columns.
        If fewer samples exist in a class, uses all available.

    Returns
    -------
    df : pandas.DataFrame
        Residue-level statistics table sorted by residue_index. Columns:

        Always present:
        - residue_index       : int, 1-based residue position
        - residue_name        : str, e.g. "100K"
        - CCR_index           : int, CCR identifier (0 if not in any CCR)
        - CCR_range           : str, e.g. "100-105" ("N/A" if not in CCR)
        - relative_importance : float, log2(rf_score / baseline)
        - contact_enhance     : float, max - min of class above_ratios (0-100%)
        - class1_cp_max       : float, mean of per-sample cp_max over class 1
        - class1_cp_top3_sum  : float, mean of per-sample cp_top{cp_raw_top_num}_sum over class 1
        - class1_cp_max_n     : float, mean of top count_n_top samples by cp_max in class 1
        - class1_cp_top3_sum_n: float, mean of top count_n_top samples by cp_top3_sum in class 1
        - class2_*, class3_*  : same columns for class 2 and class 3

        Only when has_ref=True:
        - ref_cp_max          : float, mean cp_max over unique sgRNA on-targets
        - ref_cp_top3_sum     : float, mean cp_top_sum over unique sgRNA on-targets
        - ref_cp_max_n        : float, mean of top count_n_top unique sgRNAs by cp_max
        - ref_cp_top3_sum_n   : float, mean of top count_n_top unique sgRNAs by cp_top_sum

    Notes
    -----
    cp_top3_sum is computed by taking the global top cp_raw_top_num values from
    the entire raw CP vector (all regions combined), not per-region.

    The _n columns rank samples within each class by the metric value (descending)
    and average the top count_n_top. This highlights residues that have very high
    contact in the most strongly interacting samples.

    Ref statistics use unique_on_targets_array which is already deduplicated
    (one entry per unique sgRNA name), avoiding bias from repeated on-target
    structures shared across multiple off-target samples.

    Examples
    --------
    >>> df = stats_residue_cp_by_type(
    ...     data_list, keep_residues, models, cas9_sequence, ccr_df,
    ...     protein_name="Cas9", verbose=True
    ... )
    >>> # Find residues with high CP in class 3 but low in class 1
    >>> candidates = df[df['class3_cp_max'] > 0.5][df['class1_cp_max'] < 0.2]
    """

    if verbose:
        print(f"\n{'='*70}")
        print("RESIDUE CP STATISTICS BY CLASS")
        print(f"{'='*70}")
        print(f"Protein: {protein_name}")
        print(f"has_ref: {has_ref}")
        print(f"count_n_top: {count_n_top}")

    if cpu_threads is None:
        cpu_threads = max(1, multiprocessing.cpu_count() - 2)

    start_time = datetime.now()

    # ------------------------------------------------------------------
    # Step 1: Preprocess data (same logic as residue_level_stats_parallel)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1: Preprocessing data...")

    sgRNA_to_index = {}
    unique_on_targets_list = []

    for data_dict in data_list:
        n_samples = len(data_dict[target_key])
        for sample_idx in range(n_samples):
            sgRNA_name, _ = data_dict['key_info'][sample_idx]
            if sgRNA_name not in sgRNA_to_index:
                sgRNA_to_index[sgRNA_name] = len(sgRNA_to_index)
                if has_ref:
                    unique_on_targets_list.append(data_dict['cp_raw_on_cas_nuc'][sample_idx])

    unique_on_targets_array = np.array(unique_on_targets_list) if has_ref else None

    all_cp_off_data = []
    all_y_labels = []
    all_sgRNA_indices = []

    for data_dict in data_list:
        n_samples = len(data_dict[target_key])
        for sample_idx in range(n_samples):
            sgRNA_name, _ = data_dict['key_info'][sample_idx]
            all_cp_off_data.append(data_dict['cp_raw_cas_nuc'][sample_idx])
            all_y_labels.append(data_dict[target_key][sample_idx])
            all_sgRNA_indices.append(sgRNA_to_index[sgRNA_name])

    all_cp_off_data = np.array(all_cp_off_data)
    all_y_labels = np.array(all_y_labels) - 1  # Convert to 0-based
    all_sgRNA_indices = np.array(all_sgRNA_indices)

    unique_labels = np.unique(all_y_labels)
    kept_indices = np.where(keep_residues)[0]

    if verbose:
        print(f"  Total samples: {len(all_y_labels)}")
        print(f"  Unique sgRNAs: {len(sgRNA_to_index)}")
        print(f"  Kept residues: {len(kept_indices)}")
        class_dist = np.bincount(all_y_labels)
        for i, cnt in enumerate(class_dist):
            print(f"  Class_{i+1}: {cnt} samples")

    # ------------------------------------------------------------------
    # Step 2: RF importance + relative_importance
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2: Calculating RF importance...")

    residue_rf_importance = calculate_residue_rf_importance(
        models, keep_residues,
        cp_raw_feature_count=cp_raw_feature_count,
        cp_diff_feature_count=cp_diff_feature_count
    )

    all_rf_scores = np.array([v['rf_score'] for v in residue_rf_importance.values()])
    percentile_threshold = np.percentile(all_rf_scores, bottom_percentile)
    bottom_scores = all_rf_scores[all_rf_scores <= percentile_threshold]
    baseline_importance = np.mean(bottom_scores) if len(bottom_scores) > 0 else np.min(all_rf_scores)
    if baseline_importance == 0:
        baseline_importance = 1e-10

    relative_importance_map = {}
    for pos_0based, info in residue_rf_importance.items():
        raw_score = max(info['rf_score'], 1e-10)
        relative_importance_map[pos_0based] = float(np.log2(raw_score / baseline_importance))

    # ------------------------------------------------------------------
    # Step 3: contact_enhance via residue_level_stats_parallel
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3: Calculating contact_enhance...")

    residue_stats_results = residue_level_stats_parallel(
        data_list=data_list,
        keep_residues=keep_residues,
        models=models,
        cpu_threads=cpu_threads,
        protein_name=protein_name,
        diff_threshold=diff_threshold,
        query_region_list=query_region_list,
        top_n=top_n,
        target_key=target_key,
        verbose=False,
        has_ref=has_ref,
        cp_raw_feature_count=cp_raw_feature_count,
        cp_diff_feature_count=cp_diff_feature_count,
        cp_raw_top_num=cp_raw_top_num
    )
    residue_stats = residue_stats_results['residue_stats']

    # Build contact_enhance map
    contact_enhance_map = {}
    for pos_0based, stats in residue_stats.items():
        class_stats = stats.get('class_stats', {})
        ratios = [class_stats[cn]['above_ratio'] for cn in sorted(class_stats.keys())]
        ce = (max(ratios) - min(ratios)) * 100.0 if len(ratios) >= 2 else 0.0
        contact_enhance_map[pos_0based] = round(ce, 2)

    # ------------------------------------------------------------------
    # Step 4: CCR mapping
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 4: Mapping residues to CCRs...")

    position_to_ccr = {}
    for _, row in ccr_ranking_df.iterrows():
        ccr_id = row['CCR_ID']
        ccr_range = row['CCR_range']
        ccr_all_residues = row['CCR_all_residues']
        if pd.notna(ccr_all_residues) and ccr_all_residues != "":
            for item in ccr_all_residues.split(','):
                item = item.strip()
                if item:
                    pos_str = ''.join(filter(str.isdigit, item))
                    if pos_str:
                        position_to_ccr[int(pos_str)] = {
                            'CCR_index': ccr_id,
                            'CCR_range': ccr_range
                        }

    # ------------------------------------------------------------------
    # Step 5: Parallel compute per-residue CP stats
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 5: Computing per-residue CP stats ({len(kept_indices)} residues)...")

    cp_stats_list = Parallel(n_jobs=cpu_threads, backend='loky', verbose=1 if verbose else 0)(
        delayed(_compute_residue_cp_stats)(
            residue_pos,
            all_cp_off_data,
            all_y_labels,
            unique_labels,
            unique_on_targets_array,
            all_sgRNA_indices,
            has_ref,
            count_n_top,
            cp_raw_top_num
        ) for residue_pos in kept_indices
    )

    # Index by residue_pos for easy lookup
    cp_stats_map = {r['residue_pos']: r for r in cp_stats_list}

    # ------------------------------------------------------------------
    # Step 6: Assemble DataFrame
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 6: Assembling DataFrame...")

    data_rows = []

    for pos_0based in kept_indices:
        position_1based = pos_0based + 1

        # Amino acid
        if position_1based <= len(protein_sequence):
            aa = protein_sequence[position_1based - 1]
            residue_name = f"{position_1based}{aa}"
        else:
            residue_name = f"{position_1based}X"

        # CCR
        if position_1based in position_to_ccr:
            ccr_index = position_to_ccr[position_1based]['CCR_index']
            ccr_range = position_to_ccr[position_1based]['CCR_range']
        else:
            ccr_index = 0
            ccr_range = "N/A"

        row = {
            'residue_index': position_1based,
            'residue_name': residue_name,
            'CCR_index': ccr_index,
            'CCR_range': ccr_range,
            'relative_importance': round(relative_importance_map.get(pos_0based, 0.0), 4),
            'contact_enhance': contact_enhance_map.get(pos_0based, 0.0),
        }

        # Per-class CP stats
        cp_s = cp_stats_map.get(pos_0based, {})
        for class_idx in unique_labels:
            class_name = f"class{class_idx + 1}"
            for col in ['cp_max', 'cp_top3_sum', 'cp_max_n', 'cp_top3_sum_n']:
                key = f"{class_name}_{col}"
                row[key] = round(cp_s.get(key, np.nan), 6) if not np.isnan(cp_s.get(key, np.nan)) else np.nan

        # Ref CP stats (only when has_ref=True)
        if has_ref:
            for col in ['ref_cp_max', 'ref_cp_top3_sum', 'ref_cp_max_n', 'ref_cp_top3_sum_n']:
                row[col] = round(cp_s.get(col, np.nan), 6) if not np.isnan(cp_s.get(col, np.nan)) else np.nan

        # ------------------------------------------------------------------
        # Delta logit columns (appended last)
        # Using cp_max (all-sample mean): delta_logit_*
        # Using cp_max_n (top-N mean):    delta_logit_n_*
        # ------------------------------------------------------------------
        c1_max   = cp_s.get('class1_cp_max',   np.nan)
        c2_max   = cp_s.get('class2_cp_max',   np.nan)
        c3_max   = cp_s.get('class3_cp_max',   np.nan)
        c1_max_n = cp_s.get('class1_cp_max_n', np.nan)
        c2_max_n = cp_s.get('class2_cp_max_n', np.nan)
        c3_max_n = cp_s.get('class3_cp_max_n', np.nan)

        # class vs class delta logit (cp_max)
        row['delta_logit_c3_c2'] = round(_safe_logit(c3_max) - _safe_logit(c2_max), 6)
        row['delta_logit_c3_c1'] = round(_safe_logit(c3_max) - _safe_logit(c1_max), 6)
        row['delta_logit_c2_c1'] = round(_safe_logit(c2_max) - _safe_logit(c1_max), 6)

        # class vs class delta logit (cp_max_n)
        row['delta_logit_n_c3_c2'] = round(_safe_logit(c3_max_n) - _safe_logit(c2_max_n), 6)
        row['delta_logit_n_c3_c1'] = round(_safe_logit(c3_max_n) - _safe_logit(c1_max_n), 6)
        row['delta_logit_n_c2_c1'] = round(_safe_logit(c2_max_n) - _safe_logit(c1_max_n), 6)

        # class vs ref delta logit (only when has_ref=True)
        if has_ref:
            ref_max   = cp_s.get('ref_cp_max',   np.nan)
            ref_max_n = cp_s.get('ref_cp_max_n', np.nan)

            row['delta_logit_c3_ref']   = round(_safe_logit(c3_max)   - _safe_logit(ref_max),   6)
            row['delta_logit_c2_ref']   = round(_safe_logit(c2_max)   - _safe_logit(ref_max),   6)
            row['delta_logit_c1_ref']   = round(_safe_logit(c1_max)   - _safe_logit(ref_max),   6)

            row['delta_logit_n_c3_ref'] = round(_safe_logit(c3_max_n) - _safe_logit(ref_max_n), 6)
            row['delta_logit_n_c2_ref'] = round(_safe_logit(c2_max_n) - _safe_logit(ref_max_n), 6)
            row['delta_logit_n_c1_ref'] = round(_safe_logit(c1_max_n) - _safe_logit(ref_max_n), 6)

        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df = df.sort_values('residue_index').reset_index(drop=True)

    if verbose:
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total residues: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Completed in {total_time:.2f} seconds")
        print(f"{'='*70}")

    return df
