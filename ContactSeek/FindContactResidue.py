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
    above a specified threshold across all samples.
    
    Parameters
    ----------
    cp_raw_data_list : list of numpy.ndarray
        List containing cp_raw_cas_nuc data for all samples.
        Each element has shape (n_residues, 18).
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
    all 18 dimensions and all samples, then applies the threshold filter.
    
    Examples
    --------
    >>> cp_raw_list = [np.random.rand(1368, 18) for _ in range(100)]
    >>> keep = filter_cas_residues(cp_raw_list, min_cp_threshold=0.15)
    >>> print(f"Kept {np.sum(keep)} out of {len(keep)} residues")
    """
    # Automatically detect protein length
    if len(cp_raw_data_list) == 0:
        raise ValueError("cp_raw_data_list is empty")
    
    n_residues = cp_raw_data_list[0].shape[0]
    max_contacts_per_residue = np.zeros(n_residues)
    
    # Iterate through all samples
    for cp_raw_matrix in cp_raw_data_list:
        # cp_raw_matrix shape: (n_residues, 18)
        # For each amino acid, find its maximum value across 18 dimensions
        max_contacts_in_sample = np.max(cp_raw_matrix, axis=1)
        
        # Update global maximum
        max_contacts_per_residue = np.maximum(max_contacts_per_residue, 
                                              max_contacts_in_sample)
    
    # Filter based on threshold
    keep_residues = max_contacts_per_residue >= min_cp_threshold
    
    if verbose:
        n_kept = np.sum(keep_residues)
        print(f"Contact probability threshold: {min_cp_threshold}")
        print(f"{protein_name} protein - Total residues: {n_residues}")
        print(f"Found {n_kept} contact residues ({n_kept/n_residues*100:.1f}%)")
    
    return keep_residues


def filter_cas_residues_by_diff(cp_diff_data_list, min_diff_threshold=0.05, verbose=True, protein_name="Cas"):
    """
    Filter Cas amino acid residues with small diff changes based on absolute diff values.
    
    This function identifies residues that show significant changes between off-target
    and on-target contact probabilities.
    
    Parameters
    ----------
    cp_diff_data_list : list of numpy.ndarray
        List containing cp_diff_cas_nuc data for all samples.
        Each element has shape (n_residues, 6).
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
    all 6 dimensions and all samples, then applies the threshold filter.
    
    Examples
    --------
    >>> cp_diff_list = [np.random.randn(1368, 6) * 0.1 for _ in range(100)]
    >>> keep = filter_cas_residues_by_diff(cp_diff_list, min_diff_threshold=0.1)
    >>> print(f"Kept {np.sum(keep)} residues with significant changes")
    """
    # Automatically detect protein length
    if len(cp_diff_data_list) == 0:
        raise ValueError("cp_diff_data_list is empty")
        
    n_residues = cp_diff_data_list[0].shape[0]
    max_abs_diff_per_residue = np.zeros(n_residues)
    
    # Iterate through all samples
    for cp_diff_matrix in cp_diff_data_list:
        # cp_diff_matrix shape: (n_residues, 6)
        # For each amino acid, find its maximum absolute diff across 6 dimensions
        max_abs_diff_in_sample = np.max(np.abs(cp_diff_matrix), axis=1)
        
        # Update global maximum
        max_abs_diff_per_residue = np.maximum(max_abs_diff_per_residue, 
                                             max_abs_diff_in_sample)
    
    # Filter based on threshold
    keep_residues = max_abs_diff_per_residue >= min_diff_threshold
    
    if verbose:
        n_kept = np.sum(keep_residues)
        print(f"\nDiff threshold: {min_diff_threshold}")
        print(f"{protein_name} protein - Found {n_kept} residues with significant changes ({n_kept/n_residues*100:.1f}%)")
    
    return keep_residues


def combine_filters(keep_residues_raw, keep_residues_diff, verbose=True):
    """
    Combine two filtering conditions, retaining only residues that satisfy both.
    
    This function performs an intersection (AND operation) of two boolean filters,
    ensuring that only residues passing both criteria are retained.
    
    Parameters
    ----------
    keep_residues_raw : numpy.ndarray
        Boolean array from raw contact probability filtering.
    keep_residues_diff : numpy.ndarray
        Boolean array from diff absolute value filtering.
    verbose : bool, default=True
        If True, print detailed combination statistics.
        
    Returns
    -------
    keep_residues_combined : numpy.ndarray
        Combined boolean array after applying both filters.
        
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
    Combined filtering:
    Raw filter kept: 3
    Diff filter kept: 3
    Both filters kept: 2 (40.0%)
    Overlap ratio: 66.7%
    """
    keep_residues_combined = keep_residues_raw & keep_residues_diff
    
    if verbose:
        n_raw = np.sum(keep_residues_raw)
        n_diff = np.sum(keep_residues_diff)
        n_combined = np.sum(keep_residues_combined)
        n_total = len(keep_residues_raw)
        
        print(f"\nIntersection of both criteria:")
        print(f"Found by raw CP: {n_raw}")
        print(f"Found by delta CP: {n_diff}")
        print(f"Found by both: {n_combined} ({n_combined/n_total*100:.1f}%)")
        print(f"Overlap: {n_combined/min(n_raw, n_diff)*100:.1f}%")
    
    return keep_residues_combined


def cluster_keep(keep_residues, window_size=5, threshold_ratio=0.8, verbose=True):
    """
    Fill isolated non-kept residues surrounded by kept residues using a sliding window approach.
    
    This function applies a sliding window to identify isolated False values that are
    surrounded by True values, and converts them to True to maintain continuity.
    
    Parameters
    ----------
    keep_residues : numpy.ndarray
        Boolean array of length n_residues, where True indicates retained residues
        and False indicates filtered residues.
    window_size : int, default=5
        Size of the sliding window (must be odd number).
    threshold_ratio : float, default=0.8
        Ratio of True values required in the window (excluding center) to convert
        the center position to True. Default 0.8 means 80%, requiring at least 4
        True values for window_size=5.
    verbose : bool, default=True
        If True, print detailed clustering information and examples.
        
    Returns
    -------
    new_keep_residues : numpy.ndarray
        Updated boolean array after filling isolated gaps.
        
    Raises
    ------
    ValueError
        If window_size is not an odd number.
        
    Notes
    -----
    - The function does not modify edge residues (within half window size from boundaries)
    - Only processes positions that are currently False in the original array
    - Provides examples of filled gaps when verbose=True
    
    Examples
    --------
    >>> keep = np.array([True, True, False, True, True, False, False])
    >>> filled = cluster_keep(keep, window_size=5, threshold_ratio=0.8)
    Cluster-based filtering with window size 5:
      Threshold: 4/4 (80%)
      Original kept residues: 4
      New kept residues: 5
      Added residues: 1
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd number")
    
    # Copy original array to avoid modifying input
    new_keep_residues = keep_residues.copy()
    original_keep_residues = keep_residues.copy()  # Save original values for comparison
    
    # Calculate minimum required True count
    min_true_count = int(np.ceil((window_size - 1) * threshold_ratio))
    
    # Window radius
    half_window = window_size // 2
    
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
def find_contact_residues(cp_raw_data_list, cp_diff_data_list, 
                         min_cp_threshold=0.15, 
                         min_diff_threshold=0.1,
                         use_cluster=True,
                         cluster_window_size=5,
                         cluster_threshold_ratio=0.8,
                         verbose=False,
                         protein_name="Cas"):
    """
    Identify contact residues through a multi-step filtering and clustering pipeline.
    
    This function integrates four key steps to identify amino acid residues that
    interact with nucleic acids:
    1. Filter by raw contact probability
    2. Filter by contact probability difference (off-target vs on-target)
    3. Combine both filters (intersection)
    4. (Optional) Fill isolated gaps using cluster-based approach
    
    Parameters
    ----------
    cp_raw_data_list : list of numpy.ndarray
        List containing cp_raw_cas_nuc data for all samples.
        Each element has shape (n_residues, 18).
    cp_diff_data_list : list of numpy.ndarray
        List containing cp_diff_cas_nuc data for all samples.
        Each element has shape (n_residues, 6).
    min_cp_threshold : float, default=0.15
        Minimum contact probability threshold for raw filtering.
    min_diff_threshold : float, default=0.1
        Minimum absolute diff value threshold for diff filtering.
    use_cluster : bool, default=True
        Whether to apply cluster-based gap filling after combining filters.
    cluster_window_size : int, default=5
        Window size for cluster-based filtering (must be odd).
    cluster_threshold_ratio : float, default=0.8
        Ratio of True values required in window for gap filling.
    verbose : bool, default=True
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
        If cp_raw_data_list or cp_diff_data_list is empty.
        If cluster_window_size is not an odd number.
        
    Notes
    -----
    The filtering pipeline ensures that only residues meeting both criteria
    (sufficient contact probability AND significant difference) are retained.
    The optional clustering step helps maintain spatial continuity by filling
    isolated gaps surrounded by kept residues.
    
    Examples
    --------
    >>> # Example 1: Basic usage with default parameters
    >>> cp_raw_list = [np.random.rand(1368, 18) for _ in range(100)]
    >>> cp_diff_list = [np.random.randn(1368, 6) * 0.1 for _ in range(100)]
    >>> keep = find_contact_residues(cp_raw_list, cp_diff_list)
    >>> print(f"Final kept residues: {np.sum(keep)}")
    
    >>> # Example 2: Without clustering
    >>> keep = find_contact_residues(
    ...     cp_raw_list, cp_diff_list, 
    ...     use_cluster=False,
    ...     verbose=True
    ... )
    
    >>> # Example 3: Custom thresholds and silent mode
    >>> keep = find_contact_residues(
    ...     cp_raw_list, cp_diff_list,
    ...     min_cp_threshold=0.2,
    ...     min_diff_threshold=0.15,
    ...     verbose=False
    ... )
    """
    
    if verbose:
        print("="*70)
        print(f"CONTACT RESIDUE IDENTIFICATION FOR {protein_name}")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Raw CP threshold: {min_cp_threshold}")
        print(f"  Delta CP threshold: {min_diff_threshold}")
        print(f"  Use clustering: {use_cluster}")
        if use_cluster:
            print(f"  Cluster window size: {cluster_window_size}")
            print(f"  Cluster threshold ratio: {cluster_threshold_ratio}")
        print()
    
    # Step 1: Filter by raw contact probability
    if verbose:
        print("Step 1: Finding contact residues by raw CP value...")
    keep_residues_raw = filter_cas_residues(
        cp_raw_data_list, 
        min_cp_threshold=min_cp_threshold,
        verbose=verbose,
        protein_name=protein_name
    )
    
    # Step 2: Filter by diff absolute value
    if verbose:
        print("\nStep 2: Finding contact residues by delta CP...")
    keep_residues_diff = filter_cas_residues_by_diff(
        cp_diff_data_list,
        min_diff_threshold=min_diff_threshold,
        verbose=verbose,
        protein_name=protein_name
    )
    
    # Step 3: Combine filters
    if verbose:
        print("\nStep 3: Identifying residues meeting both criteria...")
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
    n_total = len(keep_residues_raw)
    n_raw = np.sum(keep_residues_raw)
    n_diff = np.sum(keep_residues_diff)
    n_combined = np.sum(keep_residues_combined)
    n_final = np.sum(keep_residues_final)
    n_added_by_cluster = n_final - n_combined if use_cluster else 0
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total residues: {n_total}")
        print(f"Found by raw CP: {n_raw} ({n_raw/n_total*100:.1f}%)")
        print(f"Found by delta CP: {n_diff} ({n_diff/n_total*100:.1f}%)")
        print(f"Found by both criteria: {n_combined} ({n_combined/n_total*100:.1f}%)")
        if use_cluster:
            print(f"After cluster refinement: {n_final} ({n_final/n_total*100:.1f}%)")
        print("="*70 + "\n")
    
    return keep_residues_final