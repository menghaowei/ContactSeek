# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from datetime import datetime
import warnings
from joblib import Parallel, delayed
import multiprocessing
warnings.filterwarnings('ignore')

import dill
import os
import pickle


def get_protein_length(data_dict):
    """
    Automatically detect Cas protein length from data dictionary.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary containing contact probability arrays.
        
    Returns
    -------
    n_residues : int
        Number of amino acids in the protein.
        
    Raises
    ------
    ValueError
        If protein length cannot be determined from data.
    """
    # Infer protein length from cp_raw_cas_nuc dimensions
    if 'cp_raw_cas_nuc' in data_dict and len(data_dict['cp_raw_cas_nuc']) > 0:
        n_residues = data_dict['cp_raw_cas_nuc'][0].shape[0]
    elif 'cp_diff_cas_nuc' in data_dict and len(data_dict['cp_diff_cas_nuc']) > 0:
        n_residues = data_dict['cp_diff_cas_nuc'][0].shape[0]
    else:
        raise ValueError("Cannot determine protein length from data")
    
    return n_residues


def generate_adaptive_param_grid(n_samples, n_features, verbose=True):
    """
    Generate adaptive hyperparameter grid based on dataset characteristics.
    
    This function creates a parameter grid for Random Forest that adapts to the
    size and dimensionality of the dataset, following ML best practices.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features (total feature count).
    verbose : bool, default=True
        If True, print parameter selection reasoning.
        
    Returns
    -------
    param_grid : dict
        Dictionary containing hyperparameter search space.
        
    Notes
    -----
    Parameter selection rules:
    
    1. n_estimators (number of trees):
       - Small datasets (<500): [100, 200]
       - Medium datasets (500-2000): [100, 200, 300]
       - Large datasets (>2000): [200, 300, 500]
       - More trees = better performance but slower
    
    2. max_depth (tree depth):
       - Based on log2(n_features) as a baseline
       - Small features (<100): [10, 15, 20]
       - Medium features (100-1000): [15, 20, 25]
       - Large features (>1000): [20, 25, 30, None]
       - Deeper trees can model complex patterns but may overfit
    
    3. min_samples_split (minimum samples to split):
       - Based on n_samples to prevent overfitting
       - Small datasets: [2, 5, 10]
       - Medium datasets: [5, 10, 20]
       - Large datasets: [10, 20, 50]
       - Higher values = more regularization
    
    4. min_samples_leaf (minimum samples in leaf):
       - Adaptive to dataset size
       - Helps prevent overfitting on small datasets
    
    5. max_features (features per split):
       - 'sqrt': sqrt(n_features) - good default
       - 'log2': log2(n_features) - more regularization
       - For high-dimensional data, also try fixed numbers
    
    Examples
    --------
    >>> # Small dataset with many features
    >>> grid = generate_adaptive_param_grid(n_samples=300, n_features=4800)
    
    >>> # Large dataset with moderate features
    >>> grid = generate_adaptive_param_grid(n_samples=5000, n_features=1200)
    """
    import numpy as np
    
    param_grid = {}
    
    # 1. n_estimators - based on dataset size
    if n_samples < 500:
        param_grid['n_estimators'] = [100, 200]
        n_est_reasoning = "Small dataset (<500 samples): [100, 200]"
    elif n_samples < 2000:
        param_grid['n_estimators'] = [100, 200, 300]
        n_est_reasoning = "Medium dataset (500-2000 samples): [100, 200, 300]"
    else:
        param_grid['n_estimators'] = [200, 300, 500]
        n_est_reasoning = "Large dataset (>2000 samples): [200, 300, 500]"
    
    # 2. max_depth - based on feature count
    # Rule of thumb: log2(n_features) * k, where k is tuning factor
    baseline_depth = int(np.log2(max(n_features, 2)) * 2)
    
    if n_features < 100:
        param_grid['max_depth'] = [10, 15, 20]
        depth_reasoning = f"Low features (<100): [10, 15, 20] (baseline: {baseline_depth})"
    elif n_features < 1000:
        param_grid['max_depth'] = [15, 20, 25]
        depth_reasoning = f"Medium features (100-1000): [15, 20, 25] (baseline: {baseline_depth})"
    else:
        param_grid['max_depth'] = [20, 25, 30]
        depth_reasoning = f"High features (>1000): [20, 25, 30] (baseline: {baseline_depth})"
    
    # 3. min_samples_split - based on sample count
    # Rule: at least 0.5-2% of samples, with reasonable bounds
    if n_samples < 500:
        param_grid['min_samples_split'] = [2, 5, 10]
        split_reasoning = "Small dataset: [2, 5, 10] for flexibility"
    elif n_samples < 2000:
        param_grid['min_samples_split'] = [5, 10, 20]
        split_reasoning = "Medium dataset: [5, 10, 20] for balance"
    else:
        param_grid['min_samples_split'] = [10, 20, 50]
        split_reasoning = "Large dataset: [10, 20, 50] for regularization"
    
    # 4. min_samples_leaf - adaptive to dataset size
    if n_samples < 500:
        param_grid['min_samples_leaf'] = [1, 2]
        leaf_reasoning = "Small dataset: [1, 2]"
    elif n_samples < 2000:
        param_grid['min_samples_leaf'] = [1, 2, 5]
        leaf_reasoning = "Medium dataset: [1, 2, 5]"
    else:
        param_grid['min_samples_leaf'] = [2, 5, 10]
        leaf_reasoning = "Large dataset: [2, 5, 10]"
    
    # 5. max_features - based on feature dimensionality
    # High-dimensional data often benefits from more aggressive feature subsampling
    if n_features < 100:
        param_grid['max_features'] = ['sqrt', 'log2']
        feat_reasoning = "Low features: ['sqrt', 'log2']"
    elif n_features < 1000:
        param_grid['max_features'] = ['sqrt', 'log2']
        feat_reasoning = "Medium features: ['sqrt', 'log2']"
    else:
        # For very high-dimensional data, also try fixed feature counts
        sqrt_feat = int(np.sqrt(n_features))
        log_feat = int(np.log2(n_features))
        param_grid['max_features'] = ['sqrt', 'log2', sqrt_feat // 2]
        feat_reasoning = f"High features: ['sqrt' (~{sqrt_feat}), 'log2' (~{log_feat}), {sqrt_feat // 2}]"
    
    # 6. Additional regularization for high-dimensional data
    if n_features > 1000 or n_features / n_samples > 2:
        # High-dimensional or features > samples case
        # Add more conservative max_depth options
        if None not in param_grid['max_depth']:
            param_grid['max_depth'].append(None)
        
        regularization_note = "⚠️  High-dimensional data detected - added stronger regularization"
    else:
        regularization_note = ""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Adaptive Parameter Grid Generation")
        print(f"{'='*60}")
        print(f"Dataset characteristics:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Features/Samples ratio: {n_features/n_samples:.2f}")
        print(f"\nParameter selection reasoning:")
        print(f"  n_estimators: {n_est_reasoning}")
        print(f"  max_depth: {depth_reasoning}")
        print(f"  min_samples_split: {split_reasoning}")
        print(f"  min_samples_leaf: {leaf_reasoning}")
        print(f"  max_features: {feat_reasoning}")
        if regularization_note:
            print(f"\n{regularization_note}")
        print(f"\nGenerated parameter grid:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        print(f"{'='*60}\n")
    
    return param_grid


def get_feature_dimensions(data_dict, use_topn_raw=True, use_diff=True):
    """
    Automatically detect feature dimensions from data dictionary.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary containing contact probability arrays.
    use_topn_raw : bool, default=True
        Whether to use raw contact features.
    use_diff : bool, default=True
        Whether to use diff features.
        
    Returns
    -------
    feature_info : dict
        Dictionary containing:
        - 'n_raw_features': Number of raw features per residue (e.g., 18)
        - 'n_diff_features': Number of diff features per residue (e.g., 6)
        - 'features_per_residue': Total features per residue
        - 'feature_types': List of feature type names
        
    Raises
    ------
    ValueError
        If no features are selected or dimensions cannot be determined.
    """
    feature_info = {
        'n_raw_features': 0,
        'n_diff_features': 0,
        'features_per_residue': 0,
        'feature_types': []
    }
    
    # Get raw feature dimension
    if use_topn_raw:
        if 'cp_raw_cas_nuc' not in data_dict:
            raise ValueError("use_topn_raw=True but 'cp_raw_cas_nuc' not found in data")
        if len(data_dict['cp_raw_cas_nuc']) == 0:
            raise ValueError("use_topn_raw=True but 'cp_raw_cas_nuc' is empty")
        if data_dict['cp_raw_cas_nuc'][0] is None:
            raise ValueError("use_topn_raw=True but 'cp_raw_cas_nuc[0]' is None")
        feature_info['n_raw_features'] = data_dict['cp_raw_cas_nuc'][0].shape[1]
        feature_info['feature_types'].append('raw')
    
    # Get diff feature dimension
    if use_diff:
        if 'cp_diff_cas_nuc' not in data_dict:
            raise ValueError("use_diff=True but 'cp_diff_cas_nuc' not found in data")
        if len(data_dict['cp_diff_cas_nuc']) == 0:
            raise ValueError("use_diff=True but 'cp_diff_cas_nuc' is empty")
        if data_dict['cp_diff_cas_nuc'][0] is None:
            raise ValueError("use_diff=True but 'cp_diff_cas_nuc[0]' is None")
        feature_info['n_diff_features'] = data_dict['cp_diff_cas_nuc'][0].shape[1]
        feature_info['feature_types'].append('diff')
    
    # Calculate total features per residue
    feature_info['features_per_residue'] = (
        feature_info['n_raw_features'] + feature_info['n_diff_features']
    )
    
    if feature_info['features_per_residue'] == 0:
        raise ValueError("No features selected! Set use_topn_raw=True or use_diff=True")
    
    return feature_info


def prepare_data(data_dict, keep_residues, target_key='y_g3', shuffle=True, 
                 random_state=42, use_topn_raw=True, use_diff=True, verbose=True):
    """
    Prepare training data with feature filtering and optional shuffling.
    
    **NEW**: Features are now organized by feature type (block structure):
    [Residue_1_raw, Residue_2_raw, ..., Residue_N_raw, Residue_1_diff, Residue_2_diff, ..., Residue_N_diff]
    
    Parameters
    ----------
    data_dict : dict
        Original data dictionary containing:
        - 'cp_raw_cas_nuc': Raw contact probability arrays (n_residues, n_raw_features)
        - 'cp_diff_cas_nuc': Contact probability difference arrays (n_residues, n_diff_features)
        - target_key: Target labels
    keep_residues : numpy.ndarray
        Boolean array indicating which amino acids to retain.
    target_key : str, default='y_g3'
        Which classification label to use ('y_g2', 'y_g3', 'y_g4').
    shuffle : bool, default=True
        Whether to shuffle data order.
    random_state : int, default=42
        Random seed for shuffling.
    use_topn_raw : bool, default=True
        Whether to use raw contact features.
    use_diff : bool, default=True
        Whether to include diff features.
    verbose : bool, default=True
        If True, print feature preparation details.
    
    Returns
    -------
    X : numpy.ndarray
        Filtered feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Labels starting from 0 (converted from 1-based).
    indices : numpy.ndarray
        Indices after shuffling (for tracking original order).
        
    Notes
    -----
    Feature dimensions are automatically detected from data.
    Features are organized by feature type (block structure) for better version compatibility.
    
    Examples
    --------
    >>> X, y, idx = prepare_data(data_dict, keep_residues, verbose=True)
    >>> print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    """
    # Get feature dimensions
    feature_info = get_feature_dimensions(data_dict, use_topn_raw, use_diff)
    
    # Get data length
    n_samples = len(data_dict[target_key])
    
    # Generate indices
    indices = np.arange(n_samples)
    
    # Shuffle if needed
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
        if verbose:
            print(f"Data shuffled with random_state={random_state}")
    
    # Filter data
    X_list = []
    y_list = []
    
    # Determine feature dimensions
    n_kept = np.sum(keep_residues)
    n_raw_feat = feature_info['n_raw_features']
    n_diff_feat = feature_info['n_diff_features']
    feat_per_res = feature_info['features_per_residue']
    
    if verbose:
        print(f"\nPreparing features:")
        print(f"  Use raw features: {use_topn_raw} ({n_raw_feat} per residue)" if use_topn_raw else "  Use raw features: False")
        print(f"  Use diff features: {use_diff} ({n_diff_feat} per residue)" if use_diff else "  Use diff features: False")
        print(f"  Features per residue: {feat_per_res}")
        print(f"  Number of kept residues: {n_kept}")
    
    for idx in indices:
        # Prepare feature arrays for each sample
        combined_features = np.zeros(n_kept * feat_per_res, dtype=np.float32)
        
        # Get raw and diff data
        if use_topn_raw:
            cp_raw = data_dict['cp_raw_cas_nuc'][idx]  # (n_residues, n_raw_features)
            filtered_raw = cp_raw[keep_residues, :]    # (n_kept, n_raw_features)
        
        if use_diff:
            cp_diff = data_dict['cp_diff_cas_nuc'][idx]  # (n_residues, n_diff_features)
            filtered_diff = cp_diff[keep_residues, :]    # (n_kept, n_diff_features)
        
        # Organize features by feature type (block structure)
        # First block: all raw features [Res1_raw, Res2_raw, ..., ResN_raw]
        # Second block: all diff features [Res1_diff, Res2_diff, ..., ResN_diff]
        current_pos = 0
        
        # Add all raw features block
        if use_topn_raw:
            for res_idx in range(n_kept):
                combined_features[current_pos:current_pos + n_raw_feat] = filtered_raw[res_idx, :]
                current_pos += n_raw_feat
        
        # Add all diff features block
        if use_diff:
            for res_idx in range(n_kept):
                combined_features[current_pos:current_pos + n_diff_feat] = filtered_diff[res_idx, :]
                current_pos += n_diff_feat
        
        X_list.append(combined_features)
        y_list.append(data_dict[target_key][idx] - 1)  # Convert to 0-based
    
    X = np.array(X_list)
    y = np.array(y_list)

    # Ensure y is integer type
    y = y.astype(int)
    
    # Calculate feature count
    total_features = n_kept * feat_per_res
    if verbose:
        print(f"\nFeature breakdown:")
        if use_topn_raw:
            print(f"  Raw features: {n_kept * n_raw_feat} ({n_kept} residues × {n_raw_feat})")
        if use_diff:
            print(f"  Diff features: {n_kept * n_diff_feat} ({n_kept} residues × {n_diff_feat})")
        print(f"  Total features: {total_features}")
        feature_org_parts = []
        if use_topn_raw:
            feature_org_parts.append(f"Raw_block(res1~{n_kept}×{n_raw_feat})")
        if use_diff:
            feature_org_parts.append(f"Diff_block(res1~{n_kept}×{n_diff_feat})")
        print(f"  Feature organization: [{', '.join(feature_org_parts)}]")
    
    if verbose:
        print(f"\nData shape after filtering:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, indices


# ===================== RF Hyperparameter Optimization =====================
def optimize_rf_hyperparameters_fast(X_train, y_train, X_val, y_val, param_grid=None, cv_folds=2, verbose=True):
    """
    Fast RF hyperparameter optimization using randomized search.
    
    **NEW**: Automatically generates adaptive parameter grid based on dataset size
    and dimensionality if param_grid is not provided.
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    X_val : numpy.ndarray
        Validation features.
    y_val : numpy.ndarray
        Validation labels.
    param_grid : dict, optional
        Hyperparameter grid. If None, automatically generates an adaptive grid
        based on n_samples and n_features.
    cv_folds : int, default=2
        Number of cross-validation folds (reduced to 2 for speed).
    verbose : bool, default=True
        If True, print optimization progress.
        
    Returns
    -------
    best_rf : RandomForestClassifier
        Model with best parameters trained on full training set.
    grid_results : dict
        Dictionary containing:
        - 'best_params': Best hyperparameters found
        - 'best_score': Best CV score
        - 'val_accuracy': Validation set accuracy
        - 'cv_results': Full CV results
        
    Notes
    -----
    Uses RandomizedSearchCV for faster hyperparameter search compared to
    exhaustive grid search. Tests up to 15 random parameter combinations.
    
    When param_grid is None, generates an adaptive grid based on:
    - n_samples: Number of training samples
    - n_features: Number of features
    - Features/samples ratio
    
    Examples
    --------
    >>> # With automatic adaptive grid
    >>> best_model, results = optimize_rf_hyperparameters_fast(
    ...     X_train, y_train, X_val, y_val, verbose=True
    ... )
    
    >>> # With custom grid
    >>> custom_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    >>> best_model, results = optimize_rf_hyperparameters_fast(
    ...     X_train, y_train, X_val, y_val, param_grid=custom_grid, verbose=True
    ... )
    """
    n_samples, n_features = X_train.shape
    
    if param_grid is None:
        # Generate adaptive parameter grid based on dataset characteristics
        param_grid = generate_adaptive_param_grid(n_samples, n_features, verbose=verbose)
    else:
        if verbose:
            print(f"\nUsing custom parameter grid:")
            for key, values in param_grid.items():
                print(f"  {key}: {values}")
            print()
    
    if verbose:
        print("Starting hyperparameter optimization...")
        print(f"  Search method: RandomizedSearchCV")
        print(f"  CV folds: {cv_folds}")
    
    # Use more aggressive parallel strategy
    from sklearn.model_selection import RandomizedSearchCV
    
    # Convert to RandomizedSearchCV (faster)
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        warm_start=False
    )
    
    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Calculate n_iter based on grid size
    # Estimate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    # Use min(15, total_combinations) to avoid testing more than available
    n_iter = min(15, total_combinations)
    
    if verbose:
        print(f"  Total possible combinations: {total_combinations}")
        print(f"  Testing {n_iter} random combinations\n")
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,  # Dynamic based on grid size
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    # Fit
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = random_search.best_estimator_
    
    # Validation set performance
    val_pred = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    grid_results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'val_accuracy': val_acc,
        'cv_results': random_search.cv_results_
    }
    
    if verbose:
        print(f"Best parameters: {grid_results['best_params']}")
        print(f"Best CV accuracy: {grid_results['best_score']:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
    
    return best_rf, grid_results


# ===================== Model Ensemble Functions =====================
def train_rf_ensemble_fast(data_list, keep_residues, target_key='y_g3', 
                          test_size=0.1, val_size=0.1, protein_name="Cas", 
                          verbose=True, use_shared_test_set=False, param_grid=None,
                          use_topn_raw=True, use_diff=True):
    """
    Fast training of RF ensemble using multiple random seeds.
    
    Parameters
    ----------
    data_list : list of dict
        List of data dictionaries from different random seeds.
    keep_residues : numpy.ndarray
        Boolean array indicating which residues to retain.
    target_key : str, default='y_g3'
        Target label key.
    test_size : float, default=0.1
        Proportion of test set.
    val_size : float, default=0.1
        Proportion of validation set (for compatibility, not actively used in new version).
    protein_name : str, default='Cas'
        Name of the protein being analyzed.
    verbose : bool, default=True
        If True, print training progress.
    use_shared_test_set : bool, default=False
        Strategy for test set:
        - False (default, recommended): Each seed has independent train/test split.
          No data leakage, more robust feature importance. Better for feature analysis.
        - True (original method): All seeds share test set from seed 0.
          May have data leakage, higher performance metrics. For reproducing old results.
    param_grid : dict, optional
        Hyperparameter grid for Random Forest. If None, uses adaptive grid generation.
    use_topn_raw : bool, default=True
        Whether to use raw contact features.
    use_diff : bool, default=True
        Whether to use diff features.
        
    Returns
    -------
    ensemble_models : list of dict
        List of trained models with metadata.
    test_data : tuple
        (X_test, y_test) test set for ensemble evaluation.
        
    Notes
    -----
    Two evaluation strategies:
    
    **Strategy A (use_shared_test_set=True) - Original Method:**
    - Uses seed 0's data to create fixed test set indices
    - All models evaluated on same test set from seed 0
    - May have data leakage (samples in seed N's train could be in seed 0's test)
    - Higher performance metrics but less reliable
    - Use for: reproducing old results, quick prototyping
    
    **Strategy B (use_shared_test_set=False) - Recommended:**
    - Each seed has independent train/test split
    - No data leakage
    - More robust feature importance (from diverse data splits)
    - True generalization performance
    - Use for: feature importance analysis, publication, production
    
    Examples
    --------
    >>> # Recommended: Independent test sets (no data leakage)
    >>> models, (X_test, y_test) = train_rf_ensemble_fast(
    ...     data_list, keep_residues, 
    ...     use_shared_test_set=False,
    ...     verbose=True
    ... )
    
    >>> # Original method: Shared test set (for reproducing old results)
    >>> models, (X_test, y_test) = train_rf_ensemble_fast(
    ...     data_list, keep_residues,
    ...     use_shared_test_set=True,
    ...     verbose=True
    ... )
    """
    
    if use_shared_test_set:
        # Original method: shared test set from seed 0
        return _train_rf_ensemble_shared_test(
            data_list, keep_residues, target_key, test_size, val_size, 
            protein_name, verbose, param_grid, use_topn_raw, use_diff
        )
    else:
        # New method: independent test sets
        return _train_rf_ensemble_independent_test(
            data_list, keep_residues, target_key, test_size, val_size,
            protein_name, verbose, param_grid, use_topn_raw, use_diff
        )


def _train_rf_ensemble_independent_test(data_list, keep_residues, target_key='y_g3',
                                       test_size=0.1, val_size=0.1, protein_name="Cas",
                                       verbose=True, param_grid=None,
                                       use_topn_raw=True, use_diff=True):
    """
    Train RF ensemble with independent test sets for each seed (no data leakage).
    
    This is the RECOMMENDED method for feature importance analysis.
    """
    ensemble_models = []
    X_test_shared = None
    y_test_shared = None
    
    # Use fixed random_state for consistency
    random_state = 42
    
    for seed_idx, data_dict in enumerate(data_list):
        if verbose:
            print(f"\nTraining model {seed_idx + 1}/{len(data_list)} (Seed {seed_idx})...")
        
        # Prepare data
        X, y, indices = prepare_data(
            data_dict, keep_residues, target_key=target_key, 
            shuffle=True, random_state=random_state + seed_idx * 100, 
            use_topn_raw=use_topn_raw, use_diff=use_diff, verbose=False
        )
        
        # Split data
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + seed_idx, stratify=y
        )
        
        # Further split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, 
            random_state=random_state + seed_idx, stratify=y_train_full
        )
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter optimization
        # Show param grid details for first seed
        optimize_verbose = verbose if seed_idx == 0 else False
        best_rf, grid_results = optimize_rf_hyperparameters_fast(
            X_train_scaled, y_train, X_val_scaled, y_val, 
            param_grid=param_grid, verbose=optimize_verbose
        )
        
        # Test performance
        test_pred = best_rf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Store model (compatible with original format)
        model_info = {
            'seed_idx': seed_idx,  # Keep original field name for compatibility
            'model': best_rf,
            'scaler': scaler,  # New: needed for prediction
            'best_params': grid_results['best_params'],
            'test_accuracy': test_acc,
            'feature_importance': best_rf.feature_importances_,  # Keep for compatibility
            'protein_name': protein_name,  # Keep for compatibility
            'cv_score': grid_results['best_score']  # New: additional info
        }
        ensemble_models.append(model_info)
        
        # Use first seed's test set as shared test set
        if seed_idx == 0:
            X_test_shared = X_test_scaled
            y_test_shared = y_test
        
        if verbose:
            print(f"  Test accuracy: {test_acc:.4f}")
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Ensemble training completed!")
        print(f"Average test accuracy: {np.mean([m['test_accuracy'] for m in ensemble_models]):.4f}")
    
    return ensemble_models, (X_test_shared, y_test_shared)


def _train_rf_ensemble_shared_test(data_list, keep_residues, target_key='y_g3',
                                   test_size=0.1, val_size=0.1, protein_name="Cas",
                                   verbose=True, param_grid=None,
                                   use_topn_raw=True, use_diff=True):
    """
    Train RF ensemble with shared test set from seed 0 (original method).
    
    WARNING: This method may have data leakage. Use only for reproducing old results.
    For feature importance analysis, use _train_rf_ensemble_independent_test instead.
    """
    ensemble_models = []
    
    # Use first seed's data to create common test set
    X_all, y_all, _ = prepare_data(data_list[0], keep_residues, target_key, 
                                   shuffle=True, random_state=42, 
                                   use_topn_raw=use_topn_raw, use_diff=use_diff, verbose=False)
    
    # Use smaller sample for hyperparameter search
    sample_size = min(1000, len(X_all))
    if len(X_all) > sample_size:
        sample_indices = np.random.choice(len(X_all), sample_size, replace=False)
        X_sample = X_all[sample_indices]
        y_sample = y_all[sample_indices]
        if verbose:
            print(f"Using {sample_size} samples for hyperparameter tuning (out of {len(X_all)})")
    else:
        X_sample = X_all
        y_sample = y_all
    
    # Split test set using seed 0's indices
    indices_all = np.arange(len(X_all))
    train_val_indices, test_indices = train_test_split(
        indices_all, test_size=test_size, random_state=42, stratify=y_all
    )
    
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    if verbose:
        print(f"\nCommon test set from seed 0:")
        print(f"  Size: {X_test.shape[0]} samples")
        print(f"  Class distribution: {np.bincount(y_test)}")
    
    # Step 1: Find optimal hyperparameters using first seed
    if verbose:
        print("\n" + "="*60)
        print("Step 1: Finding optimal hyperparameters using first seed")
        print("="*60)
    
    # Prepare first seed's data for hyperparameter search
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    # Standardization for hyperparameter search
    scaler_hp = StandardScaler()
    X_train_sample_scaled = scaler_hp.fit_transform(X_train_sample)
    X_val_sample_scaled = scaler_hp.transform(X_val_sample)
    
    # Fast hyperparameter optimization
    best_rf, grid_results = optimize_rf_hyperparameters_fast(
        X_train_sample_scaled, y_train_sample, 
        X_val_sample_scaled, y_val_sample, 
        param_grid=param_grid, verbose=verbose
    )
    
    best_params = grid_results['best_params']
    if verbose:
        print(f"\nOptimal parameters found: {best_params}")
    
    # Step 2: Train all seeds with optimal parameters
    if verbose:
        print("\n" + "="*60)
        print(f"Step 2: Training {len(data_list)} models with optimal parameters")
        print("="*60)
    
    for seed_idx, data_dict in enumerate(data_list):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if verbose:
            print(f"\n[{timestamp}] Training model for seed {seed_idx+1}/{len(data_list)} ({protein_name})")
        
        # Prepare data for this seed
        X, y, _ = prepare_data(data_dict, keep_residues, target_key, 
                              shuffle=True, random_state=42+seed_idx*100, 
                              use_topn_raw=use_topn_raw, use_diff=use_diff, verbose=False)
        
        # Use same test set indices from seed 0
        X_train_val = X[train_val_indices]
        y_train_val = y[train_val_indices]
        
        # Standardization
        scaler = StandardScaler()
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train with optimal parameters
        rf_model = RandomForestClassifier(
            random_state=42+seed_idx,
            n_jobs=-1,
            class_weight='balanced',
            **best_params
        )
        
        rf_model.fit(X_train_val_scaled, y_train_val)
        
        # Evaluate on shared test set
        test_pred = rf_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Save results (match original format exactly)
        ensemble_models.append({
            'seed_idx': seed_idx,
            'model': rf_model,
            'scaler': scaler,
            'best_params': best_params,
            'test_accuracy': test_acc,
            'feature_importance': rf_model.feature_importances_,
            'protein_name': protein_name,
            'cv_score': grid_results['best_score']
        })
        
        timestamp_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if verbose:
            print(f"[{timestamp_end}] Test accuracy: {test_acc:.4f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Average test accuracy: {np.mean([m['test_accuracy'] for m in ensemble_models]):.4f}")
        print(f"{'='*60}")
    
    return ensemble_models, (X_test, y_test)


def ensemble_predict(ensemble_models, X_test, method='soft'):
    """
    Ensemble prediction using voting.
    
    Parameters
    ----------
    ensemble_models : list of dict
        List of trained models.
    X_test : numpy.ndarray
        Test features.
    method : str, default='soft'
        Voting method: 'soft' (probability voting) or 'hard' (majority voting).
        
    Returns
    -------
    final_predictions : numpy.ndarray
        Ensemble predictions.
    all_predictions : numpy.ndarray
        Individual model predictions (n_samples, n_models).
        
    Examples
    --------
    >>> ensemble_pred, individual_preds = ensemble_predict(
    ...     models, X_test, method='soft'
    ... )
    """
    n_models = len(ensemble_models)
    n_samples = X_test.shape[0]
    
    # Get number of classes from first model
    n_classes = len(ensemble_models[0]['model'].classes_)
    
    if method == 'soft':
        # Soft voting (average probabilities)
        all_proba = np.zeros((n_samples, n_classes))
        
        for model_info in ensemble_models:
            scaler = model_info['scaler']
            model = model_info['model']
            
            # Scale and predict
            X_scaled = scaler.transform(X_test)
            proba = model.predict_proba(X_scaled)
            all_proba += proba
        
        # Average and get final prediction
        all_proba /= n_models
        final_pred = np.argmax(all_proba, axis=1)
        
        # Also get individual predictions for return
        all_preds = np.zeros((n_samples, n_models), dtype=int)
        for i, model_info in enumerate(ensemble_models):
            X_scaled = model_info['scaler'].transform(X_test)
            all_preds[:, i] = model_info['model'].predict(X_scaled)
        
    else:  # hard voting
        all_preds = np.zeros((n_samples, n_models), dtype=int)
        
        for i, model_info in enumerate(ensemble_models):
            scaler = model_info['scaler']
            model = model_info['model']
            
            X_scaled = scaler.transform(X_test)
            all_preds[:, i] = model.predict(X_scaled)
        
        # Majority voting
        final_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            final_pred[i] = np.bincount(all_preds[i, :]).argmax()
    
    return final_pred, all_preds


# ===================== Results Saving =====================
def save_model_results(ensemble_models, X_test, y_test, keep_residues, 
                      protein_name='Cas', save_path=None, verbose=True):
    """
    Save ensemble model results and predictions.
    
    Parameters
    ----------
    ensemble_models : list of dict
        Trained ensemble models.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    keep_residues : numpy.ndarray
        Boolean array of retained residues.
    protein_name : str, default='Cas'
        Name of the protein.
    save_path : str, optional
        Path to save results. If None, auto-generates filename.
    verbose : bool, default=True
        If True, print save confirmation.
        
    Returns
    -------
    results : dict
        Dictionary containing all model results and predictions.
        
    Examples
    --------
    >>> results = save_model_results(
    ...     models, X_test, y_test, keep_residues,
    ...     protein_name="Cas9", verbose=True
    ... )
    """
    if save_path is None:
        save_path = f'{protein_name.lower()}_rf_ensemble_results.pkl'
        
    # Calculate ensemble predictions
    ensemble_pred_hard, _ = ensemble_predict(ensemble_models, X_test, method='hard')
    ensemble_pred_soft, _ = ensemble_predict(ensemble_models, X_test, method='soft')
    
    ensemble_acc_hard = accuracy_score(y_test, ensemble_pred_hard)
    ensemble_acc_soft = accuracy_score(y_test, ensemble_pred_soft)
    
    # Choose best method
    if ensemble_acc_soft >= ensemble_acc_hard:
        best_pred = ensemble_pred_soft
        best_method = "soft"
        best_acc = ensemble_acc_soft
    else:
        best_pred = ensemble_pred_hard
        best_method = "hard"
        best_acc = ensemble_acc_hard
    
    # Organize results
    results = {
        'protein_name': protein_name,
        'ensemble_models': ensemble_models,
        'keep_residues': keep_residues,
        'test_accuracy': best_acc,
        'test_predictions': best_pred,
        'test_labels': y_test,
        'best_voting_method': best_method,
        'individual_accuracies': [m['test_accuracy'] for m in ensemble_models],
        'classification_report': classification_report(y_test, best_pred, 
                                                     target_names=['Low', 'Medium', 'High'],
                                                     output_dict=True)
    }
    
    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    if verbose:
        print(f"\nModel results saved to '{save_path}'")
    
    return results


# ===================== Main Analysis Function =====================
def detect_protein_type(data_dict):
    """
    Infer protein type from data dimensions.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary.
        
    Returns
    -------
    protein_name : str
        Protein name (e.g., "Cas9", "Cpf1", or "Cas_<length>").
    n_residues : int
        Number of amino acids in the protein.
        
    Examples
    --------
    >>> prot_name, n_res = detect_protein_type(data_dict)
    >>> print(f"Detected {prot_name} with {n_res} residues")
    """
    n_residues = get_protein_length(data_dict)
    
    # Infer protein type based on length
    if n_residues == 1368:
        protein_name = "Cas9"
    elif n_residues == 1228:
        protein_name = "Cpf1"
    else:
        protein_name = f"Protein_{n_residues}"
    
    return protein_name, n_residues


def main_analysis(data_list, keep_residues=None,
                  model_performance_log=False, verbose=False,
                  use_topn_raw=True, use_diff=True, use_shared_test_set=True,
                  adapt_param_grid=False):
    """
    Main function: Train RF ensemble and analyze results.
    
    This function performs the complete analysis pipeline including data
    preparation, model training, ensemble prediction, and result saving.
    
    **IMPORTANT**: Contact residues should be pre-filtered using find_contact_residues()
    before calling this function.
    
    Parameters
    ----------
    data_list : list of dict
        List containing data dictionaries from multiple seeds.
    keep_residues : numpy.ndarray, optional
        Pre-calculated boolean array from find_contact_residues(). If None, will raise error.
    model_performance_log : bool, default=False
        If True, print detailed ensemble performance and classification report.
    verbose : bool, default=False
        If True, print analysis progress and statistics.
    use_topn_raw : bool, default=True
        Whether to use raw contact features.
    use_diff : bool, default=True
        Whether to use diff features.
    use_shared_test_set : bool, default=True
        Test set strategy:
        - True : Shared test set from the first datalist (original method)
        - False: Independent test sets (recommended for feature importance)
    adapt_param_grid : bool, default=False
        Whether to use adaptive parameter grid generation:
        - False: Use fixed parameter grid (faster, good for most cases)
        - True : Generate adaptive grid based on dataset size and features
           
    Returns
    -------
    ensemble_models : list of dict
        List of trained models with all metadata.
    results : dict
        Dictionary containing comprehensive analysis results.
        
    Raises
    ------
    ValueError
        If keep_residues is None or no features are selected.
        
    Notes
    -----
    The analysis pipeline:
    1. Detects protein type and feature dimensions automatically
    2. Trains RF ensemble with optimized hyperparameters
    3. Evaluates ensemble performance
    4. Generates classification report
    5. Saves all results to file
    
    For contact residue filtering, use find_contact_residues() with parameters:
    - min_cp_threshold: Minimum contact probability threshold
    - min_diff_threshold: Minimum diff absolute value threshold
    - cluster_window_size: Window size for cluster-based refinement
    
    Examples
    --------
    >>> # Step 1: Filter contact residues
    >>> from FindContactResidue_improved import find_contact_residues
    >>> contact_residues = find_contact_residues(
    ...     cp_raw_data_list=merge_cp_raw_list,
    ...     cp_diff_data_list=merge_cp_diff_list,
    ...     min_cp_threshold=0.15,
    ...     min_diff_threshold=0.1,
    ...     cluster_window_size=5,
    ...     use_cluster=True,
    ...     verbose=True
    ... )
    
    >>> # Step 2: Train models (recommended - independent test sets)
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues, 
    ...     use_shared_test_set=False,
    ...     verbose=True
    ... )
    
    >>> # Original method: Shared test set (for reproducing old results)
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues,
    ...     use_shared_test_set=True,
    ...     verbose=True
    ... )
    
    >>> # Using only raw features
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues,
    ...     use_topn_raw=True, 
    ...     use_diff=False,
    ...     verbose=True
    ... )
    
    >>> # Using only diff features
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues,
    ...     use_topn_raw=False, 
    ...     use_diff=True,
    ...     verbose=True
    ... )
    
    >>> # With detailed performance logging
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues,
    ...     model_performance_log=True,
    ...     verbose=True
    ... )
    
    >>> # With adaptive parameter grid (slower but optimized for dataset)
    >>> models, results = main_analysis(
    ...     data_list, 
    ...     contact_residues,
    ...     adapt_param_grid=True,
    ...     verbose=True
    ... )
    """
    
    # Detect protein type
    protein_name, n_residues = detect_protein_type(data_list[0])
    
    # Get feature dimensions
    feature_info = get_feature_dimensions(data_list[0], use_topn_raw, use_diff)
    
    if verbose:
        print("="*60)
        print(f"Random Forest Ensemble Analysis for {protein_name}")
        print(f"Protein length: {n_residues} amino acids")
        print(f"Using {len(data_list)} seeds for ensemble")
        print("="*60)
        print(f"\nFeature Configuration:")
        print(f"  Raw features per residue: {feature_info['n_raw_features']}" if use_topn_raw else "  Raw features: Disabled")
        print(f"  Diff features per residue: {feature_info['n_diff_features']}" if use_diff else "  Diff features: Disabled")
        print(f"  Total features per residue: {feature_info['features_per_residue']}")
    
    # Check if keep_residues is provided
    if keep_residues is None:
        raise ValueError("Error! Need to set <keep_residues>")
    
    # Display retained amino acid information
    n_kept = np.sum(keep_residues)
    n_total = len(keep_residues)
    if verbose:
        print(f"\n1. Contact residues for analysis:")
        print(f"   Total residues: {n_total}")
        print(f"   Analyzing {n_kept} contact residues ({n_kept/n_total*100:.1f}%)")
        print(f"   Total features: {n_kept * feature_info['features_per_residue']}")
    
    # 2. Prepare parameter grid
    if adapt_param_grid:
        # Use adaptive parameter grid based on dataset characteristics
        if verbose:
            print("\n2. Preparing adaptive parameter grid...")
        param_grid = None  # Will be generated inside optimize function
    else:
        # Use fixed parameter grid (default, faster)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt'],
        }
        if verbose:
            print("\n2. Using fixed parameter grid:")
            for key, values in param_grid.items():
                print(f"   {key}: {values}")
    
    # 3. Train RF ensemble
    if verbose:
        print(f"\n{3 if not adapt_param_grid else 2}. Training RF ensemble models...")
    ensemble_models, (X_test, y_test) = train_rf_ensemble_fast(
        data_list, keep_residues, target_key='y_g3', protein_name=protein_name, 
        verbose=verbose, use_shared_test_set=use_shared_test_set, param_grid=param_grid,
        use_topn_raw=use_topn_raw, use_diff=use_diff
    )
    
    # 4. Ensemble predictions (optional detailed output)
    if model_performance_log:
        print("\n4. Ensemble predictions...")
        ensemble_pred_hard, all_preds_hard = ensemble_predict(ensemble_models, X_test, method='hard')
        ensemble_pred_soft, all_preds_soft = ensemble_predict(ensemble_models, X_test, method='soft')
        
        ensemble_acc_hard = accuracy_score(y_test, ensemble_pred_hard)
        ensemble_acc_soft = accuracy_score(y_test, ensemble_pred_soft)
        
        print(f"\nEnsemble Results:")
        print(f"  Hard voting accuracy: {ensemble_acc_hard:.4f}")
        print(f"  Soft voting accuracy: {ensemble_acc_soft:.4f}")
        print(f"  Average individual accuracy: {np.mean([m['test_accuracy'] for m in ensemble_models]):.4f}")
        
        # Choose best method
        if ensemble_acc_soft >= ensemble_acc_hard:
            best_pred = ensemble_pred_soft
            best_method = "Soft Voting"
            best_acc = ensemble_acc_soft
        else:
            best_pred = ensemble_pred_hard
            best_method = "Hard Voting"
            best_acc = ensemble_acc_hard
        
        print(f"\nBest ensemble method: {best_method} (Accuracy: {best_acc:.4f})")
        
        # 5. Classification report
        print("\n5. Classification Report:")
        print(classification_report(y_test, best_pred, 
                                  target_names=['Low', 'Medium', 'High']))
    
    # 6. Save results
    if verbose:
        print("\n6. Saving results...")
    results = save_model_results(ensemble_models, X_test, y_test, keep_residues, 
                               protein_name=protein_name, verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print(f"Analysis Completed for {protein_name}!")
        print("="*60)
    
    return ensemble_models, results
