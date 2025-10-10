# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
import os


# ===================== Correlation Matrix Calculation =====================
def calculate_residue_correlation_matrix(data_list, keep_residues, protein_name="Cas", verbose=True):
    """
    Calculate correlation matrix between retained residues across multiple datasets.
    
    This function computes pairwise correlations between residues based on their
    contact probability profiles across all samples from multiple seeds.
    
    Parameters
    ----------
    data_list : list of dict
        List containing data dictionaries from multiple seeds.
        Each dict must contain 'y_g3' and 'cp_raw_cas_nuc' keys.
    keep_residues : numpy.ndarray
        Boolean array indicating which residues to retain for analysis.
    protein_name : str, default="Cas"
        Protein name for display purposes.
    verbose : bool, default=True
        If True, print detailed progress and statistics.
        
    Returns
    -------
    corr_matrix : numpy.ndarray
        Correlation matrix of shape (n_kept, n_kept) where n_kept is the
        number of retained residues.
    kept_indices : numpy.ndarray
        Array of indices for retained residues (0-based).
        
    Notes
    -----
    - Automatically detects protein length from data
    - NaN values in correlation matrix are replaced with 0
    - Matrix is symmetrized and diagonal is set to 1.0
    - Uses contact probability profiles (18 dimensions) for correlation
    
    Examples
    --------
    >>> corr_matrix, kept_idx = calculate_residue_correlation_matrix(
    ...     data_list, keep_residues, protein_name="Cas9", verbose=True
    ... )
    >>> print(f"Correlation matrix shape: {corr_matrix.shape}")
    """
    if verbose:
        print(f"Calculating residue correlation matrix for {protein_name}...")
    
    kept_indices = np.where(keep_residues)[0]
    n_kept = len(kept_indices)
    
    # Automatically detect protein length
    n_residues = len(keep_residues)
    
    if verbose:
        print(f"{protein_name} protein: {n_residues} total residues")
        print(f"Number of kept residues: {n_kept}")
    
    # Collect feature vectors for each residue
    residue_profiles = defaultdict(list)
    
    total_samples = 0
    for data_dict in data_list:
        n_samples = len(data_dict['y_g3'])
        total_samples += n_samples
        
        for sample_idx in range(n_samples):
            # Use generic key name
            cp_raw = data_dict['cp_raw_cas_nuc'][sample_idx]
            
            for res_idx in kept_indices:
                features = cp_raw[res_idx, :]
                residue_profiles[res_idx].append(features)
    
    if verbose:
        print(f"Total samples across all seeds: {total_samples}")
    
    # Build feature matrix
    feature_matrices = []
    for res_idx in kept_indices:
        res_features = np.concatenate(residue_profiles[res_idx])
        feature_matrices.append(res_features)
    
    feature_matrix = np.array(feature_matrices)
    
    if verbose:
        print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Calculate correlation matrix
    if verbose:
        print("Computing correlation matrix...")
    corr_matrix = np.corrcoef(feature_matrix)
    
    # Handle NaN values
    if np.any(np.isnan(corr_matrix)):
        if verbose:
            print("Warning: NaN values found in correlation matrix, replacing with 0")
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Ensure matrix symmetry
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    if verbose:
        print(f"Correlation matrix shape: {corr_matrix.shape}")
        print(f"Correlation range: [{np.min(corr_matrix):.3f}, {np.max(corr_matrix):.3f}]")
    
    return corr_matrix, kept_indices


# ===================== Contact Region Identification =====================
def identify_contact_regions(corr_matrix, kept_indices, 
                             correlation_threshold=0.4,
                             max_region_size=15,
                             band_width=5,
                             min_internal_corr=0.3,
                             compatible_with_rf=True,
                             verbose=True):
    """
    Identify contact regions based on correlation matrix analysis.
    
    This function detects spatially continuous groups of residues with high
    internal correlation, forming functional contact regions. Large regions
    are automatically subdivided to maintain biological interpretability.
    
    Parameters
    ----------
    corr_matrix : numpy.ndarray
        Correlation matrix of shape (n_residues, n_residues).
    kept_indices : numpy.ndarray
        Array of kept residue indices (0-based).
    correlation_threshold : float, default=0.4
        Minimum correlation required to form a region.
    max_region_size : int, default=15
        Maximum region size before subdivision is applied.
    band_width : int, default=5
        Bandwidth for diagonal band analysis.
    min_internal_corr : float, default=0.3
        Minimum average internal correlation within a region.
    compatible_with_rf : bool, default=True
        If True, output format compatible with RF importance calculations.
    verbose : bool, default=True
        If True, print detailed region detection information.
        
    Returns
    -------
    contact_regions : list of dict
        List of detected contact regions. Each dict contains:
        - 'cluster_id': Region identifier
        - 'contact_region_id': Same as cluster_id
        - 'indices': List of residue indices within the region
        - 'positions': Array of residue positions (0-based)
        - 'size': Number of residues in the region
        - 'avg_correlation': Average internal correlation
        - 'type': Region type ('contact_region', 'contact_region_subdivided', etc.)
        - 'continuous': Boolean indicating spatial continuity
        - 'super_cluster_id': Super-cluster identifier
        
    Notes
    -----
    The algorithm ensures:
    - Complete coverage of all retained residues
    - No overlapping regions
    - Biological continuity (no long-distance gaps)
    - Automatic subdivision of overly large regions
    
    Examples
    --------
    >>> regions = identify_contact_regions(
    ...     corr_matrix, kept_indices,
    ...     correlation_threshold=0.6,
    ...     max_region_size=14,
    ...     verbose=True
    ... )
    >>> print(f"Detected {len(regions)} contact regions")
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CONTACT REGION DETECTION")
        print(f"{'='*60}")
        print(f"Correlation threshold: {correlation_threshold}")
        print(f"Max region size: {max_region_size}")
        print(f"Min internal correlation: {min_internal_corr}")
        print(f"Band width: {band_width}")
    
    n_residues = len(kept_indices)
    assigned = np.zeros(n_residues, dtype=bool)
    contact_regions = []
    region_id = 0
    
    # Main loop: ensure every residue is assigned
    i = 0
    while i < n_residues:
        if assigned[i]:
            i += 1
            continue
            
        # Start a new region
        region_indices = [i]
        region_positions = [kept_indices[i]]
        
        # Try to extend region
        j = i + 1
        while j < n_residues and not assigned[j]:
            # Check continuity
            if kept_indices[j] - kept_indices[j-1] > 1:
                # Not continuous, end current region
                break
                
            # Calculate average correlation after adding new residue
            test_indices = region_indices + [j]
            avg_corr = calculate_region_correlation(test_indices, corr_matrix)
            
            # Check correlation with current region
            corr_with_region = []
            for idx in region_indices:
                corr_with_region.append(abs(corr_matrix[idx, j]))
            mean_corr_with_region = np.mean(corr_with_region)
            
            # Decide whether to add
            if avg_corr >= min_internal_corr and mean_corr_with_region >= correlation_threshold:
                region_indices.append(j)
                region_positions.append(kept_indices[j])
                j += 1
            else:
                # Insufficient correlation, end current region
                break
        
        # Check region size
        if len(region_indices) > max_region_size:
            # Need subdivision
            sub_regions = subdivide_large_region(
                region_indices, region_positions, corr_matrix, max_region_size, verbose=False
            )
            
            for sub_indices, sub_positions in sub_regions:
                avg_corr = calculate_region_correlation(sub_indices, corr_matrix)
                
                if compatible_with_rf:
                    # RF importance compatible format
                    contact_regions.append({
                        'cluster_id': region_id,
                        'contact_region_id': region_id,
                        'indices': sub_indices,
                        'positions': sub_positions,
                        'size': len(sub_indices),
                        'avg_correlation': avg_corr,
                        'type': 'contact_region_subdivided',
                        'continuous': True,
                        'super_cluster_id': region_id
                    })
                else:
                    contact_regions.append({
                        'contact_region_id': region_id,
                        'indices': sub_indices,
                        'positions': sub_positions,
                        'size': len(sub_indices),
                        'avg_correlation': avg_corr,
                        'type': 'contact_region_subdivided',
                        'continuous': True
                    })
                region_id += 1
                
                # Mark as assigned
                for idx in sub_indices:
                    assigned[idx] = True
        else:
            # Normal-sized region
            avg_corr = calculate_region_correlation(region_indices, corr_matrix)
            
            if compatible_with_rf:
                contact_regions.append({
                    'cluster_id': region_id,
                    'contact_region_id': region_id,
                    'indices': region_indices,
                    'positions': np.array(region_positions),
                    'size': len(region_indices),
                    'avg_correlation': avg_corr,
                    'type': 'contact_region',
                    'continuous': True,
                    'super_cluster_id': region_id
                })
            else:
                contact_regions.append({
                    'contact_region_id': region_id,
                    'indices': region_indices,
                    'positions': np.array(region_positions),
                    'size': len(region_indices),
                    'avg_correlation': avg_corr,
                    'type': 'contact_region',
                    'continuous': True
                })
            region_id += 1
            
            # Mark as assigned
            for idx in region_indices:
                assigned[idx] = True
        
        # Move to next unassigned position
        i = j if j < n_residues else i + 1
    
    # Ensure all residues are covered (handle any remaining singletons)
    for i in range(n_residues):
        if not assigned[i]:
            if compatible_with_rf:
                contact_regions.append({
                    'cluster_id': region_id,
                    'contact_region_id': region_id,
                    'indices': [i],
                    'positions': np.array([kept_indices[i]]),
                    'size': 1,
                    'avg_correlation': 1.0,
                    'type': 'contact_region_singleton',
                    'continuous': True,
                    'super_cluster_id': region_id
                })
            else:
                contact_regions.append({
                    'contact_region_id': region_id,
                    'indices': [i],
                    'positions': np.array([kept_indices[i]]),
                    'size': 1,
                    'avg_correlation': 1.0,
                    'type': 'contact_region_singleton',
                    'continuous': True
                })
            region_id += 1
            assigned[i] = True
    
    # Sort by position
    contact_regions.sort(key=lambda x: x['positions'][0])
    
    # Reassign IDs
    for i, region in enumerate(contact_regions):
        if compatible_with_rf:
            region['cluster_id'] = i
            region['super_cluster_id'] = i
        region['contact_region_id'] = i
    
    if verbose:
        # Print statistics
        print(f"\nFound {len(contact_regions)} contact regions")
        print(f"Coverage: {sum(r['size'] for r in contact_regions)}/{n_residues} residues")
        
        # Size distribution
        size_distribution = defaultdict(int)
        type_distribution = defaultdict(int)
        
        for region in contact_regions:
            size_distribution[region['size']] += 1
            type_distribution[region['type']] += 1
        
        print("\nSize distribution:")
        for size in sorted(size_distribution.keys()):
            print(f"  Size {size}: {size_distribution[size]} regions")
        
        print("\nType distribution:")
        for type_name, count in type_distribution.items():
            print(f"  {type_name}: {count}")
        
        # Print examples
        print("\nExample regions:")
        for i, region in enumerate(contact_regions[:10]):
            print(f"  Region {region['contact_region_id']}: "
                  f"size={region['size']}, "
                  f"range={region['positions'][0]+1}-{region['positions'][-1]+1}, "
                  f"avg_corr={region['avg_correlation']:.3f}, "
                  f"type={region['type']}")
    
    return contact_regions


def subdivide_large_region(indices, positions, corr_matrix, max_size, verbose=False):
    """
    Subdivide overly large regions into smaller functional units.
    
    Parameters
    ----------
    indices : list
        List of residue indices within the region.
    positions : list or numpy.ndarray
        List of residue positions.
    corr_matrix : numpy.ndarray
        Correlation matrix.
    max_size : int
        Maximum allowed region size.
    verbose : bool, default=False
        If True, print subdivision information.
        
    Returns
    -------
    sub_regions : list of tuple
        List of (sub_indices, sub_positions) tuples for subdivided regions.
        
    Notes
    -----
    The function finds optimal split points by identifying positions with
    minimum cross-correlation between adjacent segments.
    """
    n = len(indices)
    if n <= max_size:
        return [(indices, np.array(positions) if not isinstance(positions, np.ndarray) else positions)]
    
    # Convert positions to numpy array for slicing
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
    # Find best split point (lowest correlation position)
    min_corr = float('inf')
    best_split = max_size
    
    # Search within reasonable range
    for split_pos in range(max_size//2, min(n-max_size//2, max_size+max_size//2)):
        # Calculate correlation between left and right segments
        left_indices = indices[:split_pos]
        right_indices = indices[split_pos:]
        
        # Calculate cross-correlation
        cross_corr = []
        for i in left_indices[-3:]:  # Last 3 from left
            for j in right_indices[:3]:  # First 3 from right
                cross_corr.append(abs(corr_matrix[i, j]))
        
        if cross_corr:
            avg_cross_corr = np.mean(cross_corr)
            if avg_cross_corr < min_corr:
                min_corr = avg_cross_corr
                best_split = split_pos
    
    # Recursively subdivide
    left_sub = subdivide_large_region(
        indices[:best_split], 
        positions[:best_split], 
        corr_matrix, 
        max_size,
        verbose=verbose
    )
    
    right_sub = subdivide_large_region(
        indices[best_split:], 
        positions[best_split:], 
        corr_matrix, 
        max_size,
        verbose=verbose
    )
    
    return left_sub + right_sub


def calculate_region_correlation(indices, corr_matrix):
    """
    Calculate average internal correlation within a region.
    
    Parameters
    ----------
    indices : list
        List of residue indices within the region.
    corr_matrix : numpy.ndarray
        Correlation matrix.
        
    Returns
    -------
    avg_correlation : float
        Average pairwise correlation within the region.
        
    Notes
    -----
    For single-residue regions, returns 1.0.
    """
    if len(indices) <= 1:
        return 1.0
    
    corr_values = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            corr_values.append(abs(corr_matrix[indices[i], indices[j]]))
    
    return np.mean(corr_values) if corr_values else 1.0


def merge_adjacent_singleton_regions(contact_regions, corr_matrix, 
                                     max_gap=2, merge_corr_threshold=0.4, 
                                     max_iterations=5, verbose=True):
    """
    Merge adjacent singleton regions based on correlation and proximity.
    
    This function iteratively merges small regions (especially singletons) that
    are close in sequence and have sufficient correlation, reducing fragmentation.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact regions to be merged.
    corr_matrix : numpy.ndarray
        Correlation matrix.
    max_gap : int, default=2
        Maximum allowed gap (in positions) between regions for merging.
    merge_corr_threshold : float, default=0.4
        Minimum correlation required for merging.
    max_iterations : int, default=5
        Maximum number of merge iterations.
    verbose : bool, default=True
        If True, print detailed merge progress and statistics.
        
    Returns
    -------
    merged_regions : list of dict
        List of regions after merging.
    merge_log : list of dict
        Log of all merge operations performed.
        
    Notes
    -----
    The function:
    - Performs multiple iterations until no more merges are possible
    - Prevents merging across large gaps (maintains biological continuity)
    - Limits merged region size to avoid overly large clusters
    - Significantly reduces the number of singleton regions
    
    Examples
    --------
    >>> merged, log = merge_adjacent_singleton_regions(
    ...     contact_regions, corr_matrix,
    ...     max_gap=2, merge_corr_threshold=0.4,
    ...     verbose=True
    ... )
    >>> print(f"Reduced to {len(merged)} regions after merging")
    """
    if verbose:
        print(f"\n{'='*60}")
        print("MERGING ADJACENT SINGLETON REGIONS")
        print(f"{'='*60}")
        print(f"Max gap allowed: {max_gap}")
        print(f"Merge correlation threshold: {merge_corr_threshold}")
    
    current_regions = [r.copy() for r in contact_regions]
    all_merge_logs = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}:")
        
        # Perform one round of merging
        merged_regions, merge_log = merge_adjacent_regions_single_pass(
            current_regions, corr_matrix, max_gap, merge_corr_threshold, verbose=False
        )
        
        # If no new merges, stop iteration
        if len(merge_log) == 0:
            if verbose:
                print(f"  No more merges possible, stopping.")
            break
            
        if verbose:
            print(f"  Performed {len(merge_log)} merges")
            print(f"  Regions: {len(current_regions)} → {len(merged_regions)}")
        
        all_merge_logs.extend(merge_log)
        current_regions = merged_regions
    
    if verbose:
        # Print overall statistics
        print(f"\nOverall merge summary:")
        print(f"Original regions: {len(contact_regions)}")
        print(f"After all merges: {len(current_regions)}")
        print(f"Total merges performed: {len(all_merge_logs)}")
        
        # Recalculate size distribution
        size_distribution = defaultdict(int)
        for region in current_regions:
            size_distribution[region['size']] += 1
        
        print(f"\nFinal size distribution:")
        print(f"{'Size':<10} {'Count':<10}")
        print("-" * 20)
        
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            bar = '█' * min(count, 50)
            print(f"{size:<10} {count:<10} {bar}")
        
        # Calculate singleton reduction
        original_singletons = sum(1 for r in contact_regions if r['size'] == 1)
        final_singletons = size_distribution.get(1, 0)
        print(f"\nSingleton reduction: {original_singletons} → {final_singletons} (-{original_singletons - final_singletons})")
    
    return current_regions, all_merge_logs


def merge_adjacent_regions_single_pass(contact_regions, corr_matrix, max_gap, merge_corr_threshold, verbose=False):
    """
    Perform a single pass of adjacent region merging.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact regions.
    corr_matrix : numpy.ndarray
        Correlation matrix.
    max_gap : int
        Maximum allowed gap between regions.
    merge_corr_threshold : float
        Minimum correlation for merging.
    verbose : bool, default=False
        If True, print merge details.
        
    Returns
    -------
    merged_regions : list of dict
        Regions after one pass of merging.
    merge_log : list of dict
        Log of merges performed in this pass.
    """
    # Sort by position
    sorted_regions = sorted(contact_regions, key=lambda x: x['positions'][0])
    
    merged_regions = []
    merge_log = []
    processed = [False] * len(sorted_regions)
    
    i = 0
    while i < len(sorted_regions):
        if processed[i]:
            i += 1
            continue
            
        current = sorted_regions[i]
        merge_candidates = [current]
        merge_candidate_indices = [i]
        
        # Look forward for mergeable regions (with size limit)
        j = i + 1
        while j < len(sorted_regions) and not processed[j]:
            next_region = sorted_regions[j]
            
            # Calculate distance from last region in merge_candidates
            last_region = merge_candidates[-1]
            gap = next_region['positions'][0] - last_region['positions'][-1] - 1
            
            if gap > max_gap:
                break
            
            # Relax size limit: allow merge if both regions are not too large
            total_size = sum(r['size'] for r in merge_candidates) + next_region['size']
            if total_size > 10:  # Merged region cannot be too large
                break
            
            # Calculate correlation (average correlation with entire merge group)
            inter_corr = []
            for region in merge_candidates:
                for idx1 in region['indices']:
                    for idx2 in next_region['indices']:
                        inter_corr.append(abs(corr_matrix[idx1, idx2]))
            
            avg_corr = np.mean(inter_corr) if inter_corr else 0
            
            if avg_corr >= merge_corr_threshold:
                merge_candidates.append(next_region)
                merge_candidate_indices.append(j)
                j += 1
            else:
                break
        
        # If found mergeable regions
        if len(merge_candidates) > 1:
            # Merge
            merged_indices = []
            merged_positions = []
            
            for region in merge_candidates:
                merged_indices.extend(region['indices'])
                pos_list = region['positions'].tolist() if isinstance(region['positions'], np.ndarray) else region['positions']
                merged_positions.extend(pos_list)
            
            # Sort
            sorted_pairs = sorted(zip(merged_indices, merged_positions), key=lambda x: x[1])
            merged_indices = [p[0] for p in sorted_pairs]
            merged_positions = [p[1] for p in sorted_pairs]
            
            merged_region = {
                'contact_region_id': len(merged_regions),
                'indices': merged_indices,
                'positions': np.array(merged_positions),
                'size': len(merged_indices),
                'avg_correlation': calculate_region_correlation(merged_indices, corr_matrix),
                'type': 'contact_region_merged',
                'continuous': check_continuity(merged_positions),
                'merged_from': [r['contact_region_id'] for r in merge_candidates]
            }
            
            merged_regions.append(merged_region)
            
            # Mark as processed
            for idx in merge_candidate_indices:
                processed[idx] = True
            
            # Log merge
            merge_log.append({
                'merged_ids': [r['contact_region_id'] for r in merge_candidates],
                'new_id': merged_region['contact_region_id'],
                'positions': f"{merged_positions[0]+1}-{merged_positions[-1]+1}",
                'size': merged_region['size'],
                'gap_info': f"merged {len(merge_candidates)} regions"
            })
            
            i = j
        else:
            # No merge
            current['contact_region_id'] = len(merged_regions)
            merged_regions.append(current)
            processed[i] = True
            i += 1
    
    return merged_regions, merge_log


def check_continuity(positions):
    """
    Check if positions are spatially continuous.
    
    Parameters
    ----------
    positions : list or numpy.ndarray
        List of residue positions.
        
    Returns
    -------
    is_continuous : bool
        True if all positions are continuous (no gaps > 1).
    """
    if len(positions) <= 1:
        return True
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] > 1:
            return False
    return True


def print_all_contact_regions(contact_regions, protein_name="Cas", verbose=True):
    """
    Print detailed information for all contact regions in order.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact regions.
    protein_name : str, default="Cas"
        Protein name for display.
    verbose : bool, default=True
        If True, print detailed region information.
        
    Notes
    -----
    Prints comprehensive statistics including:
    - Total region count and residue coverage
    - Region type distribution
    - Size distribution with visualization
    - Detailed list of all regions with positions
    - Coverage verification
    - Correlation statistics
    """
    if not verbose:
        return
    
    print(f"\n{'='*80}")
    print(f"ALL CONTACT REGIONS for {protein_name}")
    print(f"{'='*80}")
    
    # Statistics
    total_regions = len(contact_regions)
    total_residues = sum(r['size'] for r in contact_regions)
    
    print(f"Total contact regions: {total_regions}")
    print(f"Total residues covered: {total_residues}")
    
    # Type statistics
    type_stats = defaultdict(int)
    for region in contact_regions:
        type_stats[region['type']] += 1
    
    print(f"\nRegion types:")
    for region_type, count in type_stats.items():
        print(f"  {region_type}: {count}")
    
    # Size distribution
    size_distribution = defaultdict(int)
    for region in contact_regions:
        size_distribution[region['size']] += 1
    
    print(f"\nSize distribution:")
    print(f"{'Size':<10} {'Count':<10} {'Bar'}")
    print("-" * 40)
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        bar = '█' * count
        print(f"{size:<10} {count:<10} {bar}")
    
    # Detailed list
    print(f"\n{'ID':<5} {'Size':<6} {'Type':<25} {'Range':<20} {'Avg.Corr':<10} {'Positions (1-based)'}")
    print("-" * 95)
    
    for region in contact_regions:
        region_id = region['contact_region_id']
        size = region['size']
        positions = region['positions']
        avg_corr = region['avg_correlation']
        region_type = region['type']
        
        # Range
        if len(positions) == 1:
            range_str = f"{positions[0]+1}"
        else:
            range_str = f"{positions[0]+1}-{positions[-1]+1}"
        
        # Position list
        if len(positions) <= 8:
            if isinstance(positions, np.ndarray):
                pos_str = ', '.join(map(str, positions + 1))
            else:
                pos_str = ', '.join(map(str, [p + 1 for p in positions]))
        else:
            # Show first 4 and last 4
            if isinstance(positions, np.ndarray):
                front = ', '.join(map(str, positions[:4] + 1))
                back = ', '.join(map(str, positions[-4:] + 1))
            else:
                front = ', '.join(map(str, [p + 1 for p in positions[:4]]))
                back = ', '.join(map(str, [p + 1 for p in positions[-4:]]))
            pos_str = f"{front}, ..., {back}"
        
        print(f"{region_id:<5} {size:<6} {region_type:<25} {range_str:<20} {avg_corr:<10.3f} {pos_str}")
    
    # Verification of coverage
    print(f"\n{'='*60}")
    print("COVERAGE VERIFICATION")
    print(f"{'='*60}")
    
    all_indices = set()
    for region in contact_regions:
        all_indices.update(region['indices'])
    
    expected_indices = set(range(sum(r['size'] for r in contact_regions)))
    if all_indices == expected_indices:
        print("✓ All residues are covered exactly once")
    else:
        missing = expected_indices - all_indices
        if missing:
            print(f"✗ Missing indices: {sorted(missing)}")
        else:
            print("✓ No missing residues")
    
    # Correlation statistics
    corr_values = [r['avg_correlation'] for r in contact_regions]
    print(f"\nCorrelation statistics:")
    print(f"  Mean: {np.mean(corr_values):.3f}")
    print(f"  Std: {np.std(corr_values):.3f}")
    print(f"  Min: {np.min(corr_values):.3f}")
    print(f"  Max: {np.max(corr_values):.3f}")
    
    # Find low correlation regions
    low_corr_regions = [r for r in contact_regions if r['avg_correlation'] < 0.3]
    if low_corr_regions:
        print(f"\nLow correlation regions (< 0.3): {len(low_corr_regions)}")
        for r in low_corr_regions[:5]:
            print(f"  Region {r['contact_region_id']}: corr={r['avg_correlation']:.3f}, "
                  f"size={r['size']}, range={r['positions'][0]+1}-{r['positions'][-1]+1}")


# ===================== Integrated Function =====================
def find_consensus_contact_regions(data_list, keep_residues,
                                   correlation_threshold=0.6,
                                   max_region_size=14,
                                   band_width=7,
                                   min_internal_corr=0.35,
                                   merge_singletons=True,
                                   max_gap=2,
                                   merge_corr_threshold=0.4,
                                   max_merge_iterations=10,
                                   protein_name="Cas",
                                   verbose=False):
    """
    Identify consensus contact regions through correlation analysis and merging.
    
    This function integrates the complete contact region identification pipeline:
    1. Calculate residue correlation matrix across all samples
    2. Identify initial contact regions based on correlation
    3. (Optional) Merge adjacent singleton regions to reduce fragmentation
    
    Parameters
    ----------
    data_list : list of dict
        List containing data dictionaries from multiple seeds.
        Each dict must contain 'y_g3' and 'cp_raw_cas_nuc' keys.
    keep_residues : numpy.ndarray
        Boolean array indicating which residues to analyze.
    correlation_threshold : float, default=0.4
        Minimum correlation required to form a region.
    max_region_size : int, default=15
        Maximum region size before automatic subdivision.
    band_width : int, default=5
        Bandwidth for diagonal band analysis.
    min_internal_corr : float, default=0.3
        Minimum average internal correlation within a region.
    merge_singletons : bool, default=True
        Whether to merge adjacent singleton regions.
    max_gap : int, default=2
        Maximum gap allowed between regions for merging.
    merge_corr_threshold : float, default=0.4
        Minimum correlation required for merging regions.
    max_merge_iterations : int, default=5
        Maximum number of merge iterations.
    protein_name : str, default="Cas"
        Protein name for display purposes.
    verbose : bool, default=True
        If True, print detailed progress and statistics.
        
    Returns
    -------
    corr_matrix : numpy.ndarray
        Correlation matrix of shape (n_kept, n_kept).
    kept_indices : numpy.ndarray
        Array of indices for retained residues (0-based).
    contact_regions : list of dict
        List of identified contact regions. Each dict contains:
        - 'cluster_id': Region identifier
        - 'contact_region_id': Same as cluster_id
        - 'indices': List of residue indices within the region
        - 'positions': Array of residue positions (0-based)
        - 'size': Number of residues in the region
        - 'avg_correlation': Average internal correlation
        - 'type': Region type
        - 'continuous': Boolean indicating spatial continuity
        - 'super_cluster_id': Super-cluster identifier
        
    Notes
    -----
    The complete pipeline ensures:
    - All retained residues are assigned to exactly one region
    - Regions maintain biological continuity
    - Overly large regions are automatically subdivided
    - Singleton regions can be merged to reduce fragmentation
    
    Examples
    --------
    >>> # Example 1: Basic usage with default parameters
    >>> corr_mat, kept_idx, regions = find_consensus_contact_regions(
    ...     data_list, keep_residues, protein_name="Cas9", verbose=True
    ... )
    >>> print(f"Found {len(regions)} contact regions")
    
    >>> # Example 2: Stricter parameters without merging
    >>> corr_mat, kept_idx, regions = find_consensus_contact_regions(
    ...     data_list, keep_residues,
    ...     correlation_threshold=0.6,
    ...     max_region_size=14,
    ...     merge_singletons=False,
    ...     verbose=True
    ... )
    
    >>> # Example 3: Silent mode for production
    >>> corr_mat, kept_idx, regions = find_consensus_contact_regions(
    ...     data_list, keep_residues,
    ...     protein_name="Cas9",
    ...     verbose=False
    ... )
    """
    
    if verbose:
        print("="*70)
        print(f"CONSENSUS CONTACT REGION IDENTIFICATION FOR {protein_name}")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Correlation threshold: {correlation_threshold}")
        print(f"  Max region size: {max_region_size}")
        print(f"  Min internal correlation: {min_internal_corr}")
        print(f"  Band width: {band_width}")
        print(f"  Merge singletons: {merge_singletons}")
        if merge_singletons:
            print(f"  Max gap for merging: {max_gap}")
            print(f"  Merge correlation threshold: {merge_corr_threshold}")
            print(f"  Max merge iterations: {max_merge_iterations}")
        print()
    
    # Step 1: Calculate correlation matrix
    if verbose:
        print("Step 1: Calculating residue correlation matrix...")
    corr_matrix, kept_indices = calculate_residue_correlation_matrix(
        data_list,
        keep_residues,
        protein_name=protein_name,
        verbose=verbose
    )
    
    # Step 2: Identify initial contact regions
    if verbose:
        print("\nStep 2: Identifying contact regions from correlation patterns...")
    contact_regions = identify_contact_regions(
        corr_matrix,
        kept_indices,
        correlation_threshold=correlation_threshold,
        max_region_size=max_region_size,
        band_width=band_width,
        min_internal_corr=min_internal_corr,
        compatible_with_rf=True,
        verbose=verbose
    )
    
    # Step 3: Optional merging of singleton regions
    if merge_singletons:
        if verbose:
            print("\nStep 3: Merging adjacent singleton regions...")
        contact_regions, merge_log = merge_adjacent_singleton_regions(
            contact_regions,
            corr_matrix,
            max_gap=max_gap,
            merge_corr_threshold=merge_corr_threshold,
            max_iterations=max_merge_iterations,
            verbose=verbose
        )
    else:
        if verbose:
            print("\nStep 3: Skipping singleton merging (merge_singletons=False)")
    
    # Print final summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total residues analyzed: {len(keep_residues)}")
        print(f"Retained residues: {np.sum(keep_residues)}")
        print(f"Final contact regions: {len(contact_regions)}")
        print(f"Residues covered: {sum(r['size'] for r in contact_regions)}")
        
        # Size distribution summary
        size_dist = defaultdict(int)
        for r in contact_regions:
            size_dist[r['size']] += 1
        
        print(f"\nRegion size distribution:")
        print(f"  Singletons: {size_dist.get(1, 0)}")
        print(f"  Size 2-5: {sum(size_dist[s] for s in range(2, 6) if s in size_dist)}")
        print(f"  Size 6-10: {sum(size_dist[s] for s in range(6, 11) if s in size_dist)}")
        print(f"  Size >10: {sum(size_dist[s] for s in size_dist if s > 10)}")
        
        print("="*70 + "\n")
    
    return corr_matrix, kept_indices, contact_regions