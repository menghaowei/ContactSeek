# -*- coding: UTF-8 -*-

import numpy as np
import os


# ===================== Data Processing Functions =====================
def get_protein_length(data_dict):
    """
    Automatically detect Cas protein length from data dictionary.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary containing contact probability arrays
        
    Returns
    -------
    n_residues : int
        Number of amino acids in the protein
        
    Raises
    ------
    ValueError
        If protein length cannot be determined from data
    """
    # Infer protein length from cp_raw_cas_nuc dimensions
    if 'cp_raw_cas_nuc' in data_dict and len(data_dict['cp_raw_cas_nuc']) > 0:
        n_residues = data_dict['cp_raw_cas_nuc'][0].shape[0]
    elif 'cp_diff_cas_nuc' in data_dict and len(data_dict['cp_diff_cas_nuc']) > 0:
        n_residues = data_dict['cp_diff_cas_nuc'][0].shape[0]
    else:
        raise ValueError("Cannot determine protein length from data")
    
    return n_residues


def filter_cas_residues(cp_raw_data_list, min_cp_threshold=0.1, verbose=True, protein_name="Cas"):
    """
    Filter Cas amino acid residues with weak nucleic acid chain interactions.
    
    This function identifies and retains only residues that have contact probability
    above a specified threshold across all samples. Automatically adapts to any 
    data dimensions (e.g., (n_residues, 18), (n_residues, 3), etc.).
    
    Parameters
    ----------
    cp_raw_data_list : list of numpy.ndarray
        List containing cp_raw_cas_nuc data for all samples.
        Each element has shape (n_residues, n_features), where n_features can be any value.
    min_cp_threshold : float, default=0.1
        Minimum contact probability threshold for retaining residues.
    verbose : bool, default=True
        If True, print detailed filtering information.
    protein_name : str, default="Cas"
        Protein name for display purposes.
    
    Returns
    -------
    keep_residues : numpy.ndarray
        Boolean array of length n_residues, where True indicates the residue
        should be retained.
        
    Raises
    ------
    ValueError
        If cp_raw_data_list is empty.
    
    Notes
    -----
    The function finds the maximum contact probability for each residue across
    all dimensions and all samples, then applies the threshold filter.
    Works with any number of features (columns) in the input data.
    
    Examples
    --------
    >>> # For Cas9 with 18 features
    >>> cp_raw_list = [np.random.rand(1368, 18) for _ in range(100)]
    >>> keep = filter_cas_residues(cp_raw_list, min_cp_threshold=0.15)
    >>> print(f"Kept {np.sum(keep)} out of {len(keep)} residues")
    
    >>> # For TadA8e with 3 features
    >>> cp_raw_list = [np.random.rand(167, 3) for _ in range(100)]
    >>> keep = filter_cas_residues(cp_raw_list, min_cp_threshold=0.15)
    """
    # Automatically detect protein length
    if len(cp_raw_data_list) == 0:
        raise ValueError("cp_raw_data_list is empty")
    
    n_residues = cp_raw_data_list[0].shape[0]
    n_features = cp_raw_data_list[0].shape[1]
    max_contacts_per_residue = np.zeros(n_residues)
    
    # Iterate through all samples
    for cp_raw_matrix in cp_raw_data_list:
        # cp_raw_matrix shape: (n_residues, n_features)
        # For each amino acid, find its maximum value across all features
        max_contacts_in_sample = np.max(cp_raw_matrix, axis=1)
        
        # Update global maximum
        max_contacts_per_residue = np.maximum(max_contacts_per_residue, 
                                              max_contacts_in_sample)
    
    # Filter based on threshold
    keep_residues = max_contacts_per_residue >= min_cp_threshold
    
    if verbose:
        n_kept = np.sum(keep_residues)
        print(f"Contact probability threshold: {min_cp_threshold}")
        print(f"{protein_name} protein - Total residues: {n_residues}, Features: {n_features}")
        print(f"Found {n_kept} contact residues ({n_kept/n_residues*100:.1f}%)")
    
    return keep_residues


def filter_cas_residues_by_diff(cp_diff_data_list, min_diff_threshold=0.05, verbose=True, protein_name="Cas"):
    """
    Filter Cas amino acid residues with small diff changes based on absolute diff values.
    
    This function identifies residues that show significant changes between off-target
    and on-target contact probabilities. Automatically adapts to any data dimensions.
    
    Parameters
    ----------
    cp_diff_data_list : list of numpy.ndarray
        List containing cp_diff_cas_nuc data for all samples.
        Each element has shape (n_residues, n_features), where n_features can be any value.
    min_diff_threshold : float, default=0.05
        Minimum absolute diff value threshold for retaining residues.
    verbose : bool, default=True
        If True, print detailed filtering information.
    protein_name : str, default="Cas"
        Protein name for display purposes.
    
    Returns
    -------
    keep_residues : numpy.ndarray
        Boolean array of length n_residues, where True indicates the residue
        should be retained.
        
    Raises
    ------
    ValueError
        If cp_diff_data_list is empty.
    
    Notes
    -----
    The function finds the maximum absolute diff value for each residue across
    all dimensions and all samples, then applies the threshold filter.
    Works with any number of features (columns) in the input data.
    
    Examples
    --------
    >>> # For Cas9 with 6 features
    >>> cp_diff_list = [np.random.randn(1368, 6) * 0.1 for _ in range(100)]
    >>> keep = filter_cas_residues_by_diff(cp_diff_list, min_diff_threshold=0.1)
    
    >>> # For other proteins with different dimensions
    >>> cp_diff_list = [np.random.randn(167, 3) * 0.1 for _ in range(100)]
    >>> keep = filter_cas_residues_by_diff(cp_diff_list, min_diff_threshold=0.1)
    """
    # Automatically detect protein length
    if len(cp_diff_data_list) == 0:
        raise ValueError("cp_diff_data_list is empty")
        
    n_residues = cp_diff_data_list[0].shape[0]
    n_features = cp_diff_data_list[0].shape[1]
    max_abs_diff_per_residue = np.zeros(n_residues)
    
    # Iterate through all samples
    for cp_diff_matrix in cp_diff_data_list:
        # cp_diff_matrix shape: (n_residues, n_features)
        # For each amino acid, find its maximum absolute diff across all features
        max_abs_diff_in_sample = np.max(np.abs(cp_diff_matrix), axis=1)
        
        # Update global maximum
        max_abs_diff_per_residue = np.maximum(max_abs_diff_per_residue, 
                                             max_abs_diff_in_sample)
    
    # Filter based on threshold
    keep_residues = max_abs_diff_per_residue >= min_diff_threshold
    
    if verbose:
        n_kept = np.sum(keep_residues)
        print(f"\nDiff threshold: {min_diff_threshold}")
        print(f"{protein_name} protein - Total residues: {n_residues}, Features: {n_features}")
        print(f"Found {n_kept} residues with significant changes ({n_kept/n_residues*100:.1f}%)")
    
    return keep_residues


def combine_filters(keep_residues_raw, keep_residues_diff, verbose=True):
    """
    Combine two filtering conditions, retaining only residues that satisfy both.
    
    This function performs an intersection (AND operation) of two boolean filters,
    ensuring that only residues passing both criteria are retained.
    
    Parameters
    ----------
    keep_residues_raw : numpy.ndarray or None
        Boolean array from raw contact probability filtering, or None to skip.
    keep_residues_diff : numpy.ndarray or None
        Boolean array from diff absolute value filtering, or None to skip.
    verbose : bool, default=True
        If True, print detailed combination statistics.
        
    Returns
    -------
    keep_residues_combined : numpy.ndarray
        Combined boolean array after applying both filters.
        If one filter is None, returns the other filter.
        If both are None, raises ValueError.
        
    Raises
    ------
    ValueError
        If both filters are None.
        
    Notes
    -----
    The function calculates and reports:
    - Number of residues kept by each individual filter
    - Number of residues kept by both filters (intersection)
    - Overlap ratio between the two filters
    
    Examples
    --------
    >>> raw_filter = np.array([True, True, False, True, False])
    >>> diff_filter = np.array([True, False, False, True, True])
    >>> combined = combine_filters(raw_filter, diff_filter)
    
    >>> # Only one filter available
    >>> combined = combine_filters(raw_filter, None)
    """
    # Handle cases where one or both filters are None
    if keep_residues_raw is None and keep_residues_diff is None:
        raise ValueError("At least one filter must be provided (both are None)")
    
    if keep_residues_raw is None:
        if verbose:
            n_diff = np.sum(keep_residues_diff)
            print(f"\nUsing only diff filter:")
            print(f"  Diff filter kept: {n_diff}")
        return keep_residues_diff
    
    if keep_residues_diff is None:
        if verbose:
            n_raw = np.sum(keep_residues_raw)
            print(f"\nUsing only raw filter:")
            print(f"  Raw filter kept: {n_raw}")
        return keep_residues_raw
    
    # Both filters provided - combine them
    keep_residues_combined = keep_residues_raw & keep_residues_diff
    
    if verbose:
        n_raw = np.sum(keep_residues_raw)
        n_diff = np.sum(keep_residues_diff)
        n_combined = np.sum(keep_residues_combined)
        n_total = len(keep_residues_raw)
        
        print(f"\nCombined filtering:")
        print(f"  Raw filter kept: {n_raw}")
        print(f"  Diff filter kept: {n_diff}")
        print(f"  Both filters kept: {n_combined} ({n_combined/n_total*100:.1f}%)")
        
        if n_raw > 0 and n_diff > 0:
            overlap_ratio = n_combined / min(n_raw, n_diff) * 100
            print(f"  Overlap ratio: {overlap_ratio:.1f}%")
    
    return keep_residues_combined


def cluster_keep(keep_residues, window_size=5, threshold_ratio=0.8, verbose=True):
    """
    Fill isolated gaps in contact residues using a sliding window approach.
    
    This function identifies False positions surrounded by True values within
    a sliding window. If the ratio of True values in the window (excluding center)
    exceeds the threshold, the center position is changed to True.
    
    Parameters
    ----------
    keep_residues : numpy.ndarray
        Boolean array indicating which residues are currently kept.
    window_size : int, default=5
        Size of sliding window (must be odd number, typically 3, 5, or 7).
    threshold_ratio : float, default=0.8
        Ratio of True values required in window to fill the gap.
        Range: (0, 1]. For example, 0.8 means 80% of positions must be True.
    verbose : bool, default=True
        If True, print detailed modification information.
        
    Returns
    -------
    new_keep_residues : numpy.ndarray
        Modified boolean array with gaps filled.
        
    Raises
    ------
    ValueError
        If window_size is not an odd number.
        If threshold_ratio is not in range (0, 1].
        
    Notes
    -----
    This function helps maintain spatial continuity by filling isolated False
    positions that are surrounded by True positions, which often represent
    genuine contact residues that were missed due to threshold effects.
    
    The window excludes the center position when counting True values, ensuring
    that the decision to fill a gap is based purely on surrounding context.
    
    Examples
    --------
    >>> keep = np.array([True, True, False, True, True, False, False, True])
    >>> new_keep = cluster_keep(keep, window_size=5, threshold_ratio=0.8)
    >>> print(new_keep)
    [True, True, True, True, True, False, False, True]
    """
    # Validate window_size
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    
    # Validate threshold_ratio
    if threshold_ratio <= 0 or threshold_ratio > 1:
        raise ValueError(f"threshold_ratio must be in range (0, 1], got {threshold_ratio}")
    
    # Calculate required True count
    half_window = window_size // 2
    positions_to_check = window_size - 1  # Exclude center position
    min_true_count = int(np.ceil(positions_to_check * threshold_ratio))
    
    # Create a copy for modification
    new_keep_residues = keep_residues.copy()
    original_keep_residues = keep_residues.copy()
    
    # Record modified positions
    modified_positions = []
    
    # Sliding window traversal (excluding edges)
    for i in range(half_window, len(keep_residues) - half_window):
        # Skip if current position is already True
        if original_keep_residues[i]:
            continue
        
        # Extract window (using original values)
        window_start = i - half_window
        window_end = i + half_window + 1
        window = original_keep_residues[window_start:window_end]
        
        # Count True values in window (excluding center position)
        true_count = np.sum(window) - window[half_window]
        
        # If condition is met, change center position to True
        if true_count >= min_true_count:
            new_keep_residues[i] = True
            modified_positions.append(i)
    
    if verbose:
        n_original = np.sum(keep_residues)
        n_new = np.sum(new_keep_residues)
        n_added = n_new - n_original
        
        print(f"\nCluster-based refinement with window size {window_size}:")
        print(f"  Threshold: {min_true_count}/{window_size-1} ({threshold_ratio*100:.0f}%)")
        print(f"  Contact residues before clustering: {n_original}")
        print(f"  Contact residues after clustering: {n_new}")
        print(f"  Residues added: {n_added}")
        
        if n_added > 0:
            print(f"  Added positions (0-based): {modified_positions[:10]}{'...' if len(modified_positions) > 10 else ''}")
            
            # Show some examples
            print("\n  Examples of filled gaps:")
            for i, pos in enumerate(modified_positions[:3]):
                window_start = max(0, pos - half_window)
                window_end = min(len(keep_residues), pos + half_window + 1)
                original_window = original_keep_residues[window_start:window_end]
                new_window = new_keep_residues[window_start:window_end]
                
                print(f"\n    Position {pos+1} (1-based):")
                print(f"      Before: {['F' if not x else 'T' for x in original_window]}")
                print(f"      After:  {['F' if not x else 'T' for x in new_window]}")
    
    return new_keep_residues


# ===================== Integrated Function =====================
def find_contact_residues(cp_raw_data_list=None, 
                         cp_diff_data_list=None, 
                         min_cp_threshold=0.15, 
                         min_diff_threshold=0.1,
                         use_cluster=True,
                         cluster_window_size=5,
                         cluster_threshold_ratio=0.8,
                         verbose=False,
                         protein_name="Cas"):
    """
    Identify contact residues through a flexible multi-step filtering pipeline.
    
    This function integrates up to four key steps to identify amino acid residues 
    that interact with nucleic acids. It's designed to work with various protein types
    and data dimensions, and can use either or both filtering criteria.
    
    Pipeline steps:
    1. (Optional) Filter by raw contact probability
    2. (Optional) Filter by contact probability difference (off-target vs on-target)
    3. Combine filters (intersection if both provided, single filter if only one)
    4. (Optional) Fill isolated gaps using cluster-based approach
    
    Parameters
    ----------
    cp_raw_data_list : list of numpy.ndarray or None, default=None
        List containing cp_raw_cas_nuc data for all samples.
        Each element has shape (n_residues, n_features).
        If None, raw contact probability filtering is skipped.
    cp_diff_data_list : list of numpy.ndarray or None, default=None
        List containing cp_diff_cas_nuc data for all samples.
        Each element has shape (n_residues, n_features).
        If None, diff filtering is skipped.
    min_cp_threshold : float, default=0.15
        Minimum contact probability threshold for raw filtering.
        Only used if cp_raw_data_list is provided.
    min_diff_threshold : float, default=0.1
        Minimum absolute diff value threshold for diff filtering.
        Only used if cp_diff_data_list is provided.
    use_cluster : bool, default=True
        Whether to apply cluster-based gap filling after combining filters.
    cluster_window_size : int, default=5
        Window size for cluster-based filtering (must be odd).
    cluster_threshold_ratio : float, default=0.8
        Ratio of True values required in window for gap filling.
    verbose : bool, default=False
        If True, print detailed information for each step.
    protein_name : str, default="Cas"
        Protein name for display purposes.
    
    Returns
    -------
    keep_residues : numpy.ndarray
        Final boolean array of length n_residues indicating which residues
        should be retained for downstream analysis.
        
    Raises
    ------
    ValueError
        If both cp_raw_data_list and cp_diff_data_list are None or empty.
        If cluster_window_size is not an odd number.
        
    Notes
    -----
    - The function automatically adapts to different protein lengths and feature dimensions
    - At least one of cp_raw_data_list or cp_diff_data_list must be provided
    - When both filters are used, only residues meeting both criteria are retained
    - The clustering step helps maintain spatial continuity by filling isolated gaps
    
    Examples
    --------
    >>> # Example 1: Cas9 with both filters (traditional use case)
    >>> cp_raw_list = [np.random.rand(1368, 18) for _ in range(100)]
    >>> cp_diff_list = [np.random.randn(1368, 6) * 0.1 for _ in range(100)]
    >>> keep = find_contact_residues(
    ...     cp_raw_data_list=cp_raw_list, 
    ...     cp_diff_data_list=cp_diff_list,
    ...     verbose=True
    ... )
    
    >>> # Example 2: TadA8e with only raw CP (no diff data available)
    >>> cp_raw_list = [np.random.rand(167, 3) for _ in range(100)]
    >>> keep = find_contact_residues(
    ...     cp_raw_data_list=cp_raw_list,
    ...     cp_diff_data_list=None,
    ...     min_cp_threshold=0.15,
    ...     verbose=True,
    ...     protein_name="TadA8e"
    ... )
    
    >>> # Example 3: Only using diff data
    >>> cp_diff_list = [np.random.randn(500, 4) * 0.1 for _ in range(100)]
    >>> keep = find_contact_residues(
    ...     cp_raw_data_list=None,
    ...     cp_diff_data_list=cp_diff_list,
    ...     min_diff_threshold=0.1,
    ...     use_cluster=False,
    ...     verbose=True
    ... )
    
    >>> # Example 4: Custom thresholds without clustering
    >>> keep = find_contact_residues(
    ...     cp_raw_data_list=cp_raw_list,
    ...     cp_diff_data_list=cp_diff_list,
    ...     min_cp_threshold=0.2,
    ...     min_diff_threshold=0.15,
    ...     use_cluster=False,
    ...     verbose=False
    ... )
    """
    
    # Validate inputs
    if (cp_raw_data_list is None or len(cp_raw_data_list) == 0) and \
       (cp_diff_data_list is None or len(cp_diff_data_list) == 0):
        raise ValueError("At least one of cp_raw_data_list or cp_diff_data_list must be provided and non-empty")
    
    # Determine protein length from available data
    if cp_raw_data_list is not None and len(cp_raw_data_list) > 0:
        n_total = cp_raw_data_list[0].shape[0]
    elif cp_diff_data_list is not None and len(cp_diff_data_list) > 0:
        n_total = cp_diff_data_list[0].shape[0]
    
    if verbose:
        print("="*70)
        print(f"CONTACT RESIDUE IDENTIFICATION FOR {protein_name}")
        print("="*70)
        print(f"\nProtein length: {n_total} residues")
        print(f"\nParameters:")
        if cp_raw_data_list is not None and len(cp_raw_data_list) > 0:
            print(f"  Raw CP threshold: {min_cp_threshold}")
            print(f"  Raw CP data shape: {cp_raw_data_list[0].shape}")
        else:
            print(f"  Raw CP filtering: DISABLED (no data provided)")
        
        if cp_diff_data_list is not None and len(cp_diff_data_list) > 0:
            print(f"  Delta CP threshold: {min_diff_threshold}")
            print(f"  Delta CP data shape: {cp_diff_data_list[0].shape}")
        else:
            print(f"  Delta CP filtering: DISABLED (no data provided)")
        
        print(f"  Use clustering: {use_cluster}")
        if use_cluster:
            print(f"  Cluster window size: {cluster_window_size}")
            print(f"  Cluster threshold ratio: {cluster_threshold_ratio}")
        print()
    
    # Step 1: Filter by raw contact probability (if data provided)
    keep_residues_raw = None
    if cp_raw_data_list is not None and len(cp_raw_data_list) > 0:
        if verbose:
            print("Step 1: Finding contact residues by raw CP value...")
        keep_residues_raw = filter_cas_residues(
            cp_raw_data_list, 
            min_cp_threshold=min_cp_threshold,
            verbose=verbose,
            protein_name=protein_name
        )
    else:
        if verbose:
            print("Step 1: Skipping raw CP filtering (no data provided)")
    
    # Step 2: Filter by diff absolute value (if data provided)
    keep_residues_diff = None
    if cp_diff_data_list is not None and len(cp_diff_data_list) > 0:
        if verbose:
            print("\nStep 2: Finding contact residues by delta CP...")
        keep_residues_diff = filter_cas_residues_by_diff(
            cp_diff_data_list,
            min_diff_threshold=min_diff_threshold,
            verbose=verbose,
            protein_name=protein_name
        )
    else:
        if verbose:
            print("\nStep 2: Skipping delta CP filtering (no data provided)")
    
    # Step 3: Combine filters
    if verbose:
        if keep_residues_raw is not None and keep_residues_diff is not None:
            print("\nStep 3: Identifying residues meeting both criteria...")
        else:
            print("\nStep 3: Using single available filter...")
    
    keep_residues_combined = combine_filters(
        keep_residues_raw, 
        keep_residues_diff,
        verbose=verbose
    )
    
    # Step 4: Optional clustering
    if use_cluster:
        if verbose:
            print("\nStep 4: Refining contact residues with cluster-based approach...")
        keep_residues_final = cluster_keep(
            keep_residues_combined,
            window_size=cluster_window_size,
            threshold_ratio=cluster_threshold_ratio,
            verbose=verbose
        )
    else:
        if verbose:
            print("\nStep 4: Skipping cluster-based refinement (use_cluster=False)")
        keep_residues_final = keep_residues_combined
    
    # Collect statistics
    n_raw = np.sum(keep_residues_raw) if keep_residues_raw is not None else 0
    n_diff = np.sum(keep_residues_diff) if keep_residues_diff is not None else 0
    n_combined = np.sum(keep_residues_combined)
    n_final = np.sum(keep_residues_final)
    n_added_by_cluster = n_final - n_combined if use_cluster else 0
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total residues: {n_total}")
        
        if keep_residues_raw is not None:
            print(f"Found by raw CP: {n_raw} ({n_raw/n_total*100:.1f}%)")
        if keep_residues_diff is not None:
            print(f"Found by delta CP: {n_diff} ({n_diff/n_total*100:.1f}%)")
        
        if keep_residues_raw is not None and keep_residues_diff is not None:
            print(f"Found by both criteria: {n_combined} ({n_combined/n_total*100:.1f}%)")
        else:
            print(f"Found by single criterion: {n_combined} ({n_combined/n_total*100:.1f}%)")
        
        if use_cluster:
            print(f"After cluster refinement: {n_final} ({n_final/n_total*100:.1f}%)")
            if n_added_by_cluster > 0:
                print(f"  (Added {n_added_by_cluster} residues via clustering)")
        print("="*70 + "\n")
    
    return keep_residues_final
