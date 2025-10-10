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


def prepare_data(data_dict, keep_residues, target_key='y_g3', shuffle=True, 
                 random_state=42, use_top3_raw=True, use_diff=True, verbose=True):
    """
    Prepare training data with feature filtering and optional shuffling.
    
    Parameters
    ----------
    data_dict : dict
        Original data dictionary containing:
        - 'cp_raw_cas_nuc': Raw contact probability arrays
        - 'cp_diff_cas_nuc': Contact probability difference arrays
        - target_key: Target labels
    keep_residues : numpy.ndarray
        Boolean array indicating which amino acids to retain.
    target_key : str, default='y_g3'
        Which classification label to use ('y_g2', 'y_g3', 'y_g4').
    shuffle : bool, default=True
        Whether to shuffle data order.
    random_state : int, default=42
        Random seed for shuffling.
    use_top3_raw : bool, default=True
        Whether to use top3 raw values (18-dimensional features).
    use_diff : bool, default=True
        Whether to include diff features (6-dimensional features).
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
    Feature dimensions per residue:
    - Raw features: 18 (6 regions × 3 top values)
    - Diff features: 6 (6 regions × 1 diff value)
    - Total: 24 features per residue
    
    Examples
    --------
    >>> X, y, idx = prepare_data(data_dict, keep_residues, verbose=True)
    >>> print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    """
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
    
    if verbose:
        print(f"\nPreparing features:")
        print(f"  Use top3 raw values: {use_top3_raw}")
        print(f"  Use diff features: {use_diff}")
        print(f"  Number of kept residues: {n_kept}")
    
    for idx in indices:
        features = []
        
        # 1. Raw contact features (top3 raw values)
        if use_top3_raw:
            cp_raw = data_dict['cp_raw_cas_nuc'][idx]  # (n_residues, 18)
            filtered_raw = cp_raw[keep_residues, :]
            features.append(filtered_raw.flatten())
        
        # 2. Diff features
        if use_diff:
            cp_diff = data_dict['cp_diff_cas_nuc'][idx]  # (n_residues, 6)
            filtered_diff = cp_diff[keep_residues, :]
            features.append(filtered_diff.flatten())
        
        # Merge features
        if len(features) > 0:
            combined_features = np.concatenate(features)
        else:
            raise ValueError("No features selected!")
        
        X_list.append(combined_features)
        y_list.append(data_dict[target_key][idx] - 1)  # Convert to 0-based
    
    X = np.array(X_list)
    y = np.array(y_list)

    # Ensure y is integer type
    y = y.astype(int)
    
    # Calculate feature count
    if use_top3_raw and use_diff:
        raw_features = n_kept * 18
        diff_features = n_kept * 6
        total_features = raw_features + diff_features
        if verbose:
            print(f"\nFeature breakdown:")
            print(f"  Raw features: {raw_features} ({n_kept} residues × 18)")
            print(f"  Diff features: {diff_features} ({n_kept} residues × 6)")
            print(f"  Total features: {total_features}")
    
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
        Hyperparameter grid. If None, uses default simplified grid.
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
    exhaustive grid search. Tests up to 10 random parameter combinations.
    
    Examples
    --------
    >>> best_model, results = optimize_rf_hyperparameters_fast(
    ...     X_train, y_train, X_val, y_val, verbose=True
    ... )
    >>> print(f"Best accuracy: {results['val_accuracy']:.4f}")
    """
    if param_grid is None:
        # Simplified parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt'],
        }
    
    if verbose:
        print("Starting FAST hyperparameter optimization...")
    
    # Use more aggressive parallel strategy
    from sklearn.model_selection import RandomizedSearchCV
    
    # Convert to RandomizedSearchCV (faster)
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced',
        warm_start=True  # Allow incremental learning
    )
    
    # Use RandomizedSearchCV instead of GridSearchCV
    n_iter = min(len(list(ParameterGrid(param_grid))), 10)  # Test at most 10 combinations
    
    random_search = RandomizedSearchCV(
        rf, param_grid, 
        n_iter=n_iter,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=1,  # RF is already parallelized
        random_state=42,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"Testing {n_iter} random parameter combinations with {cv_folds}-fold CV...")
    random_search.fit(X_train, y_train)
    
    # Evaluate on validation set
    best_rf = random_search.best_estimator_
    val_pred = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    if verbose:
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
    
    grid_results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'val_accuracy': val_acc,
        'cv_results': random_search.cv_results_
    }
    
    return best_rf, grid_results


def train_rf_ensemble_fast(data_list, keep_residues, target_key='y_g3', 
                          test_size=0.1, val_size=0.1, protein_name="Cas", verbose=True):
    """
    Train RF ensemble using fast hyperparameter optimization.
    
    This function trains multiple RF models (one per seed) with optimized
    hyperparameters and creates an ensemble for robust predictions.
    
    Parameters
    ----------
    data_list : list of dict
        List containing data dictionaries from multiple seeds.
    keep_residues : numpy.ndarray
        Boolean array indicating which amino acid positions to retain.
    target_key : str, default='y_g3'
        Target label key in data dictionaries.
    test_size : float, default=0.1
        Proportion of data for test set.
    val_size : float, default=0.1
        Proportion of data for validation set (used in hyperparameter tuning).
    protein_name : str, default="Cas"
        Protein name for display purposes.
    verbose : bool, default=True
        If True, print training progress and statistics.
        
    Returns
    -------
    ensemble_models : list of dict
        List of trained models. Each dict contains:
        - 'seed_idx': Seed index
        - 'model': Trained RandomForestClassifier
        - 'best_params': Best hyperparameters used
        - 'test_accuracy': Test set accuracy
        - 'feature_importance': Feature importance array
        - 'protein_name': Protein name
    test_data : tuple
        (X_test, y_test) for final ensemble evaluation.
        
    Notes
    -----
    The function:
    1. Uses first seed to find optimal hyperparameters
    2. Applies these parameters to train models on all seeds
    3. Uses a common test set across all seeds for fair comparison
    
    Examples
    --------
    >>> models, (X_test, y_test) = train_rf_ensemble_fast(
    ...     data_list, keep_residues, protein_name="Cas9", verbose=True
    ... )
    >>> print(f"Trained {len(models)} ensemble models")
    """
    ensemble_models = []
    
    # Use first seed's data to create common test set
    X_all, y_all, _ = prepare_data(data_list[0], keep_residues, target_key, 
                                   shuffle=True, random_state=42, verbose=False)
    
    # Use smaller sample for hyperparameter search
    sample_size = min(1000, len(X_all))
    if len(X_all) > sample_size:
        sample_indices = np.random.choice(len(X_all), sample_size, replace=False)
        X_sample = X_all[sample_indices]
        y_sample = y_all[sample_indices]
        if verbose:
            print(f"\nUsing {sample_size} samples for hyperparameter tuning (out of {len(X_all)})")
    else:
        X_sample = X_all
        y_sample = y_all
    
    # Split test set
    indices_all = np.arange(len(X_all))
    train_val_indices, test_indices = train_test_split(
        indices_all, test_size=test_size, random_state=42, stratify=y_all
    )
    
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    if verbose:
        print(f"\nCommon test set size: {X_test.shape[0]} samples")
        print(f"Test set class distribution: {np.bincount(y_test)}")
    
    # Step 1: Find optimal hyperparameters using first seed
    if verbose:
        print("\n" + "="*60)
        print("Step 1: Finding optimal hyperparameters using first seed")
        print("="*60)
    
    # Prepare first seed's data
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    # Fast hyperparameter optimization
    best_rf, grid_results = optimize_rf_hyperparameters_fast(
        X_train_sample, y_train_sample, X_val_sample, y_val_sample, verbose=verbose
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
        
        # Prepare data
        X, y, _ = prepare_data(data_dict, keep_residues, target_key, 
                              shuffle=True, random_state=42+seed_idx*100, verbose=False)
        
        # Use same test set indices
        X_train_val = X[train_val_indices]
        y_train_val = y[train_val_indices]
        
        # Train with optimal parameters
        rf_model = RandomForestClassifier(
            random_state=42+seed_idx,
            n_jobs=-1,
            class_weight='balanced',
            **best_params
        )
        
        rf_model.fit(X_train_val, y_train_val)
        
        # Evaluate
        test_pred = rf_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Save results
        ensemble_models.append({
            'seed_idx': seed_idx,
            'model': rf_model,
            'best_params': best_params,
            'test_accuracy': test_acc,
            'feature_importance': rf_model.feature_importances_,
            'protein_name': protein_name
        })
        
        timestamp_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if verbose:
            print(f"[{timestamp_end}] Completed seed {seed_idx+1}/{len(data_list)}")
    
    return ensemble_models, (X_test, y_test)


def ensemble_predict(ensemble_models, X, method='hard'):
    """
    Make predictions using ensemble of models.
    
    Parameters
    ----------
    ensemble_models : list of dict
        List of model dictionaries containing trained classifiers.
    X : numpy.ndarray
        Input features for prediction.
    method : str, default='hard'
        Voting method: 'hard' for hard voting, 'soft' for soft voting.
        
    Returns
    -------
    ensemble_pred : numpy.ndarray
        Ensemble predictions.
    individual_preds : numpy.ndarray
        Individual model predictions (for hard voting) or
        probability predictions (for soft voting).
        
    Notes
    -----
    Hard voting: Takes majority vote from individual predictions.
    Soft voting: Averages predicted probabilities and takes argmax.
    
    Examples
    --------
    >>> hard_pred, _ = ensemble_predict(models, X_test, method='hard')
    >>> soft_pred, _ = ensemble_predict(models, X_test, method='soft')
    """
    predictions = []
    
    for model_dict in ensemble_models:
        rf_model = model_dict['model']
        if method == 'hard':
            pred = rf_model.predict(X)
        else:
            pred = rf_model.predict_proba(X)
        predictions.append(pred)
    
    if method == 'hard':
        # Hard voting
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 0, predictions
        )
    else:
        # Soft voting
        predictions = np.mean(predictions, axis=0)
        ensemble_pred = np.argmax(predictions, axis=1)
    
    return ensemble_pred, np.array(predictions)


def save_model_results(ensemble_models, X_test, y_test, keep_residues, 
                      protein_name="Cas", save_path=None, verbose=True):
    """
    Save model results to file.
    
    Parameters
    ----------
    ensemble_models : list of dict
        List of trained models.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels.
    keep_residues : numpy.ndarray
        Boolean array of kept residues.
    protein_name : str, default="Cas"
        Protein name.
    save_path : str, optional
        Save file path. If None, auto-generates filename.
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
        Protein name (e.g., "Cas9", "Cpf1").
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
        protein_name = f"Cas_{n_residues}"
    
    return protein_name, n_residues


def main_analysis(data_list, keep_residues=None, use_fast_mode=True, 
                  min_cp_threshold=0.15, min_diff_threshold=0.1,
                  cluster_window_size=5, model_performance_log=False, verbose=False):
    """
    Main function: Train RF ensemble and analyze results.
    
    This function performs the complete analysis pipeline including data
    preparation, model training, ensemble prediction, and result saving.
    
    Parameters
    ----------
    data_list : list of dict
        List containing data dictionaries from multiple seeds.
    keep_residues : numpy.ndarray, optional
        Pre-calculated boolean array. If None, must be provided.
    use_fast_mode : bool, default=True
        Whether to use fast mode (currently always True).
    min_cp_threshold : float, default=0.15
        Minimum contact probability threshold (for reference only).
    min_diff_threshold : float, default=0.1
        Minimum diff absolute value threshold (for reference only).
    cluster_window_size : int, default=5
        Cluster window size (for reference only).
    model_performance_log : bool, default=False
        If True, print detailed ensemble performance and classification report.
    verbose : bool, default=True
        If True, print analysis progress and statistics.
        
    Returns
    -------
    ensemble_models : list of dict
        List of trained models with all metadata.
    results : dict
        Dictionary containing comprehensive analysis results.
        
    Raises
    ------
    ValueError
        If keep_residues is None.
        
    Notes
    -----
    The analysis pipeline:
    1. Detects protein type automatically
    2. Trains RF ensemble with optimized hyperparameters
    3. Evaluates ensemble performance
    4. Generates classification report
    5. Saves all results to file
    
    Examples
    --------
    >>> # Basic usage
    >>> models, results = main_analysis(
    ...     data_list, keep_residues, verbose=True
    ... )
    
    >>> # With detailed performance logging
    >>> models, results = main_analysis(
    ...     data_list, keep_residues,
    ...     model_performance_log=True,
    ...     verbose=True
    ... )
    
    >>> # Silent mode
    >>> models, results = main_analysis(
    ...     data_list, keep_residues, verbose=False
    ... )
    """
    
    # Detect protein type
    protein_name, n_residues = detect_protein_type(data_list[0])
    
    if verbose:
        print("="*60)
        print(f"Random Forest Ensemble Analysis for {protein_name}")
        print(f"Protein length: {n_residues} amino acids")
        print(f"Using {len(data_list)} seeds for ensemble")
        print(f"Mode: {'FAST' if use_fast_mode else 'STANDARD'}")
        print("="*60)
    
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
    
    # 2. Train RF ensemble
    if verbose:
        print("\n2. Training RF ensemble models...")
    ensemble_models, (X_test, y_test) = train_rf_ensemble_fast(
        data_list, keep_residues, target_key='y_g3', protein_name=protein_name, verbose=verbose
    )
    
    # 3. Ensemble predictions (optional detailed output)
    if model_performance_log:
        print("\n3. Ensemble predictions...")
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
        
        # 4. Classification report
        print("\n4. Classification Report:")
        print(classification_report(y_test, best_pred, 
                                  target_names=['Low', 'Medium', 'High']))
    
    # 5. Save results
    if verbose:
        print("\n5. Saving results...")
    results = save_model_results(ensemble_models, X_test, y_test, keep_residues, 
                               protein_name=protein_name, verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print(f"Analysis Completed for {protein_name}!")
        print("="*60)
    
    return ensemble_models, results