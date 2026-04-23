# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from scipy import stats


# =========================== contact region importance ===========================
def ranking_consensus_contact_region(contact_regions, models, keep_residues,
                                    protein_sequence=None,
                                    normalization_method='power_zscore',
                                    aggregation_method='sum-mean-topn',
                                    cp_raw_feature_count=18,
                                    cp_diff_feature_count=6,
                                    use_topn_cp_raw_feature_count=None,
                                    use_topn_cp_diff_feature_count=None,
                                    output_tsv=None,
                                    back_res_dict=False,
                                    block_residue_list=None,
                                    block_mode=False,
                                    protein_name=None,
                                    verbose=False):
    """
    Rank consensus contact regions (CCR) by RF importance and optionally export to TSV.
    
    This function computes RF importance scores for each contact region, ranks them,
    and can export detailed results to both a TSV file and pandas DataFrame with 
    comprehensive residue information.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact region dictionaries, each containing:
        - 'contact_region_id': Region identifier (0-based)
        - 'positions': Array of residue positions (0-based)
        - 'size': Number of residues in region
        - 'avg_correlation': Average internal correlation
        - 'type': Region type string
    models : list of dict
        List of trained Random Forest model dictionaries, each containing:
        - 'model': Trained sklearn RandomForestClassifier
    keep_residues : numpy.ndarray
        Boolean array indicating retained residues for analysis.
    protein_sequence : str, optional
        Complete protein sequence as single-letter amino acid string.
        If provided, used to generate complete residue annotations in output.
        Length must match protein length. Default is None.
    normalization_method : str, default='power_zscore'
        Method for score normalization. Options:
        - 'power_zscore': Power transform followed by z-score normalization
        - 'robust_minmax': Robust min-max using 5th/95th percentiles
        - 'percentile': Rank-based percentile normalization
        - 'zscore': Standard z-score with sigmoid mapping
        - 'log_transform': Log transformation normalization
    aggregation_method : str, default='sum-mean-topn'
        Method for aggregating residue scores within regions.
        Options: 'sum-mean-topn', 'sum-mean-topn-mix4', 'sum-mean-topn-mix5', 'sum-mean-topn-mix6',
        'sum-mean-topn-filterimp-max3', 'sum-mean-topn-filterimp-max4', 'sum-mean-topn-mix6-filterimp-max4'
    cp_raw_feature_count : int, default=18
        Number of raw contact probability features per residue.
    cp_diff_feature_count : int, default=6
        Number of diff contact probability features per residue.
    use_topn_cp_raw_feature_count : int, optional
        Number of top raw features to use for each residue.
        None (default) = use all; 0 = don't use raw features; N>0 = use top N.
    use_topn_cp_diff_feature_count : int, optional
        Number of top diff features to use for each residue.
        None (default) = use all; 0 = don't use diff features; N>0 = use top N.
    output_tsv : str, optional
        Path for TSV output file. If None, no file is saved. Default is None.
    back_res_dict : bool, default=False
        If True, returns tuple (df, dict). If False, returns only df.
        The dict contains complete contact_region_rf_importance results.
    block_residue_list : list of int, optional
        List of residue positions (1-based) to block. Their importance will be set to 0.
        Default is None.
    block_mode : bool, default=False
        If True, enable residue blocking functionality. If False, block_residue_list and
        protein_name are ignored.
    protein_name : str, optional
        Name of the protein for applying reference blocking positions.
        Used only when block_mode=True. Options: "Cas9", "lbCas12a", "TadA8e".
        Default is None.
    verbose : bool, default=True
        If True, prints detailed progress and ranking information.
        If False, suppresses all console output.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with CCR ranking results containing columns:
        - CCR_rank: Rank by importance (1-based, int)
        - CCR_ID: Contact region identifier (1-based, int)
        - CCR_size: Number of actual residues in the region (int)
        - CCR_range: Position range string (1-based)
        - raw_score: Raw RF importance score (float)
        - norm_score: Normalized importance score 0-1 (float)
        - CCR_all_residues: Residue annotation for ACTUAL positions in the region
        - range_all_residues: Residue annotation for ALL positions in the range
    contact_region_rf_importance : dict, optional
        Only returned if back_res_dict=True. Contains:
        - 'residue_importance': Dict of per-residue RF scores
        - 'region_importance': Dict of per-region RF scores and metadata
        - 'normalization_method': Normalization method used
        - 'aggregation_method': Aggregation method used
    
    Notes
    -----
    Residue Blocking:
    - When block_mode=True, specified residues will have their feature importances set to 0
    - block_residue_list specifies custom positions (1-based indexing)
    - protein_name specifies reference blocking positions from ref_block_dict
    - Both lists are merged and applied together
    - Blocked residues are excluded from region importance calculations
    
    Output Format (all 1-based indexing):
    - DataFrame and TSV have identical content
    - CCR_rank: Rank by importance (1 = highest)
    - CCR_ID: Contact region identifier (converted to 1-based)
    - CCR_size: Number of ACTUAL residues in the region (not range size)
    - CCR_range: Position range in format "start-end" (1-based)
    - raw_score: Raw RF importance score
    - norm_score: Normalized importance score (0-1 range)
    - CCR_all_residues: Residue annotation for actual positions in the contact region
      Format: "123A,124T,125G" (position + amino acid letter)
      Only includes positions that are actually in the region (positions array)
      For non-continuous regions, some positions in the range may be missing
    - range_all_residues: Residue annotation for complete range
      Format: "123A,124T,125G,126D,127E" (position + amino acid letter)
      Includes ALL positions from min to max, even if not in contact region
      This shows the complete range including gaps
    
    Return Behavior:
    - If back_res_dict=False: returns df
    - If back_res_dict=True: returns (df, contact_region_rf_importance)
    
    Examples
    --------
    >>> # Example 1: Basic usage without blocking
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence="MDEFGHIK...",
    ...     verbose=True
    ... )
    
    >>> # Example 2: Block custom residues
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cas9_sequence,
    ...     block_mode=True,
    ...     block_residue_list=[770, 776],
    ...     verbose=True
    ... )
    
    >>> # Example 3: Block reference positions for Cas9
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cas9_sequence,
    ...     block_mode=True,
    ...     protein_name="Cas9",
    ...     verbose=True
    ... )
    
    >>> # Example 4: Block both custom and reference positions
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cas9_sequence,
    ...     block_mode=True,
    ...     protein_name="Cas9",
    ...     block_residue_list=[770, 776],
    ...     verbose=True
    ... )
    
    See Also
    --------
    calculate_contact_region_rf_importance : Core computation function
    normalize_scores : Score normalization methods
    _export_ccr_ranking_to_tsv : TSV export helper
    _create_ccr_ranking_dataframe : DataFrame creation helper
    """
    
    # Step 1: Prepare blocked residues list
    blocked_residues_0based = None
    if block_mode:
        # Reference blocking dictionary (1-based residue index)
        ref_block_dict = {
            "Cas9": [10, 840, 1107, 1108, 1109, 1333, 1335],
            "lbCas12a": [832, 925, 1148],
            "TadA8e": [87]
        }
        
        # Collect all residues to block
        blocked_residues_1based = []
        
        # Add custom blocked residues
        if block_residue_list is not None:
            blocked_residues_1based.extend(block_residue_list)
        
        # Add reference blocked residues
        if protein_name is not None and protein_name in ref_block_dict:
            blocked_residues_1based.extend(ref_block_dict[protein_name])
        
        # Convert to 0-based and remove duplicates
        if len(blocked_residues_1based) > 0:
            blocked_residues_0based = list(set([pos - 1 for pos in blocked_residues_1based]))
            blocked_residues_0based.sort()
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"RESIDUE BLOCKING ENABLED")
                print(f"{'='*80}")
                print(f"Protein name: {protein_name if protein_name else 'N/A'}")
                print(f"Custom blocked residues (1-based): {sorted(block_residue_list) if block_residue_list else 'None'}")
                if protein_name and protein_name in ref_block_dict:
                    print(f"Reference blocked residues (1-based): {sorted(ref_block_dict[protein_name])}")
                else:
                    print(f"Reference blocked residues (1-based): None")
                print(f"Total blocked residues: {len(blocked_residues_0based)}")
                print(f"Blocked positions (1-based): {sorted([pos + 1 for pos in blocked_residues_0based])}")
                print(f"{'='*80}\n")
    
    # Step 2: Calculate contact region RF importance with blocking
    contact_region_rf_importance = calculate_contact_region_rf_importance(
        contact_regions=contact_regions,
        models=models,
        keep_residues=keep_residues,
        normalization_method=normalization_method,
        aggregation_method=aggregation_method,
        cp_raw_feature_count=cp_raw_feature_count,
        cp_diff_feature_count=cp_diff_feature_count,
        use_topn_cp_raw_feature_count=use_topn_cp_raw_feature_count,
        use_topn_cp_diff_feature_count=use_topn_cp_diff_feature_count,
        blocked_residues=blocked_residues_0based,
        verbose=verbose
    )
    
    # Step 3: Extract and sort regions
    cr_importance = contact_region_rf_importance['region_importance']
    sorted_crs = sorted(cr_importance.items(), 
                       key=lambda x: x[1]['region_score'], 
                       reverse=True)
    
    # Step 4: Print ranking table if verbose
    if verbose:
        print(f"\n{'='*110}")
        print(f"TOP CONSENSUS CONTACT REGIONS (CCR) BY RF IMPORTANCE")
        print(f"{'='*110}")
        print(f"{'Rank':<6} {'CCR_ID':<8} {'Size':<6} {'Range':<20}  "
              f"{'Score':<12} {'Corr':<8}")
        print("-" * 110)
        
        for rank, (region_id, region_info) in enumerate(sorted_crs[:5], 1):
            positions = region_info['positions']
            if len(positions) == 0:
                pos_range = "N/A"
            elif len(positions) == 1:
                pos_range = str(positions[0])
            else:
                pos_range = f"{min(positions)}-{max(positions)}"
            
            print(f"{rank:<6} {region_info['region_id']+1:<8} {region_info['size']:<6} "
                  f"{pos_range:<20}"
                  f"{region_info['normalized_score']:<12.4f} "
                  f"{region_info['avg_correlation']:<8.3f}")
        
        print(f"{'='*110}\n")
    
    # Step 5: Create DataFrame
    df = _create_ccr_ranking_dataframe(
        sorted_crs=sorted_crs,
        protein_sequence=protein_sequence
    )
    
    # Step 6: Export to TSV if requested
    if output_tsv is not None:
        df.to_csv(output_tsv, sep='\t', index=False)
        if verbose:
            print(f"✓ CCR ranking exported to: {output_tsv}")
            print(f"  Total regions: {len(df)}")
            print(f"  Format: TSV with {len(df.columns)} columns (1-based indexing)\n")
    
    # Step 7: Return based on back_res_dict flag
    if back_res_dict:
        return df, contact_region_rf_importance
    else:
        return df


def _create_ccr_ranking_dataframe(sorted_crs, protein_sequence=None):
    """
    Create a pandas DataFrame from sorted contact region results.
    
    Parameters
    ----------
    sorted_crs : list of tuple
        List of (region_id, region_info) tuples sorted by importance.
    protein_sequence : str, optional
        Complete protein sequence. If provided, generates detailed residue annotations.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with CCR ranking information (1-based indexing).
    """
    df_data = []
    
    for rank, (region_id, region_info) in enumerate(sorted_crs, 1):
        positions = region_info['positions']
        
        # Calculate actual CCR size based on actual positions
        actual_ccr_size = len(positions)
        
        # Calculate position range (1-based)
        if len(positions) == 0:
            pos_range = "N/A"
            pos_min, pos_max = None, None
        elif len(positions) == 1:
            pos_range = str(positions[0] + 1)
            pos_min = pos_max = positions[0] + 1
        else:
            pos_min = min(positions) + 1
            pos_max = max(positions) + 1
            pos_range = f"{pos_min}-{pos_max}"
        
        # Generate residue annotations based on ACTUAL positions in the region
        if protein_sequence is not None and len(positions) > 0:
            # CCR_all_residues: Only actual positions in the contact region
            residue_annotations = []
            for pos in positions:  # 0-based positions from contact region
                if pos < len(protein_sequence):
                    aa = protein_sequence[pos]
                    residue_annotations.append(f"{pos + 1}{aa}")  # 1-based for output
                else:
                    residue_annotations.append(f"{pos + 1}X")  # X for unknown
            ccr_all_residues = ",".join(residue_annotations)
            
            # range_all_residues: All positions in the range (even if not in contact region)
            if pos_min is not None and pos_max is not None:
                range_annotations = []
                for pos in range(pos_min - 1, pos_max):  # 0-based for indexing
                    if pos < len(protein_sequence):
                        aa = protein_sequence[pos]
                        range_annotations.append(f"{pos + 1}{aa}")  # 1-based for output
                    else:
                        range_annotations.append(f"{pos + 1}X")  # X for unknown
                range_all_residues = ",".join(range_annotations)
            else:
                range_all_residues = "N/A"
        else:
            ccr_all_residues = "N/A"
            range_all_residues = "N/A"
        
        df_data.append({
            'CCR_rank': rank,
            'CCR_ID': region_info['region_id'] + 1,  # Convert to 1-based
            'CCR_size': actual_ccr_size,  # Actual number of residues in the region
            'CCR_range': pos_range,
            'raw_score': region_info['region_score'],
            'norm_score': region_info['normalized_score'],
            'CCR_all_residues': ccr_all_residues,  # Only actual residues in region
            'range_all_residues': range_all_residues  # All residues in the range
        })
    
    df = pd.DataFrame(df_data)
    return df


# ===================== Core RF Importance Calculation =====================
def calculate_contact_region_rf_importance(contact_regions, models, keep_residues,
                                          normalization_method='power_zscore',
                                          aggregation_method='sum-mean-topn',
                                          cp_raw_feature_count=18,
                                          cp_diff_feature_count=6,
                                          use_topn_cp_raw_feature_count=None,
                                          use_topn_cp_diff_feature_count=None,
                                          blocked_residues=None,
                                          verbose=False):
    """
    Calculate Random Forest importance scores for contact regions with flexible feature aggregation.
    
    This function computes residue-level and region-level importance scores from an ensemble
    of Random Forest models, supporting flexible feature selection and multiple aggregation methods.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact region dictionaries. Each dict contains:
        - 'contact_region_id': Region identifier (0-based)
        - 'positions': Array of residue positions (0-based) in this region
        - 'size': Number of residues in the region
        - 'avg_correlation': Average correlation within region (optional)
    models : list of dict
        List of trained Random Forest model dictionaries. Each dict contains:
        - 'model': Trained sklearn RandomForestClassifier with feature_importances_
        - 'kept_feature_indices': Indices of features used in training
    keep_residues : numpy.ndarray
        Boolean array indicating which residues were retained for analysis.
        Shape: (n_residues,)
    normalization_method : str, default='power_zscore'
        Method for normalizing region scores to 0-1 range. Options:
        - 'power_zscore': Power transform + z-score + sigmoid (recommended)
        - 'robust_minmax': Robust min-max using 5th/95th percentiles
        - 'percentile': Rank-based percentile normalization
        - 'zscore': Z-score + sigmoid mapping
        - 'log_transform': Log transform + min-max scaling
    aggregation_method : str, default='sum-mean-topn'
        Method for aggregating feature importances within regions:
        - 'sum-mean-topn': Original adaptive top-N based on region size
        - 'sum-mean-topn-mix4': 1 res→use 1, 2 res→sum 2, 3 res→top 3, ≥4 res→top 4
        - 'sum-mean-topn-mix5': 1 res→use 1, 2 res→sum 2, 3-4 res→top 3, ≥5 res→top 4
        - 'sum-mean-topn-mix6': 1 res→use 1, 2 res→sum 2, 3-5 res→top 3, ≥6 res→top 4
        - 'sum-mean-topn-filterimp-max3': Filter by mean/median, max 3 residues
        - 'sum-mean-topn-filterimp-max4': Filter by mean/median, max 4 residues
        - 'sum-mean-topn-mix6-filterimp-max4': Adaptive max with filtering
    cp_raw_feature_count : int, default=18
        Number of raw contact probability features per residue.
        These features correspond to cp_raw_cas_nuc in the model.
    cp_diff_feature_count : int, default=6
        Number of diff contact probability features per residue.
        These features correspond to cp_diff_cas_nuc in the model.
    use_topn_cp_raw_feature_count : int, optional
        Number of top raw features to select per residue for importance calculation.
        - None (default): Use all cp_raw_feature_count features
        - 0: Don't use raw features at all (selected_raw_importance = 0)
        - N (>0): Select top N raw features by importance
    use_topn_cp_diff_feature_count : int, optional
        Number of top diff features to select per residue for importance calculation.
        - None (default): Use all cp_diff_feature_count features
        - 0: Don't use diff features at all (selected_diff_importance = 0)
        - N (>0): Select top N diff features by importance
    blocked_residues : list of int, optional
        List of residue positions (0-based) to block. Feature importances for these
        residues will be set to 0. Default is None.
    verbose : bool, default=False
        If True, prints detailed progress information and statistics.
    
    Returns
    -------
    importance_dict : dict
        Dictionary containing:
        - 'residue_importance': dict
            Keys are residue positions (0-based), values are dicts with:
            - 'mean_importance': Mean importance across models
            - 'std_importance': Standard deviation across models
            - 'raw_importances': List of raw importance per model
        - 'region_importance': dict
            Keys are region IDs (0-based), values are dicts with:
            - 'region_id': Region identifier
            - 'positions': Array of residue positions in region
            - 'size': Number of residues
            - 'residue_scores': List of importance scores for residues
            - 'residue_positions': List of residue positions
            - 'region_score': Aggregated region importance score
            - 'normalized_score': Normalized score (0-1 range)
            - 'top_residues_idx': Indices of top residues used
            - 'top_residues_pos': Positions of top residues used
            - 'aggregation_method': Method used for aggregation
            - 'n_top_used': Number of top residues used
            - 'avg_correlation': Average correlation in region
        - 'normalization_method': Normalization method used
        - 'aggregation_method': Aggregation method used
    
    Notes
    -----
    Feature Layout (BLOCK STRUCTURE):
    **CRITICAL**: Features are organized in BLOCK structure, NOT interleaved!
    
    The model was trained with features organized as:
    [All Raw features | All Diff features]
    
    Structure:
    - Raw block: All residues' raw features concatenated
      [Residue_1_raw(18), Residue_2_raw(18), ..., Residue_N_raw(18)]
    - Diff block: All residues' diff features concatenated
      [Residue_1_diff(6), Residue_2_diff(6), ..., Residue_N_diff(6)]
    
    Total features = n_kept_residues * cp_raw_feature_count + n_kept_residues * cp_diff_feature_count
    
    Example with 3 kept residues (cp_raw=18, cp_diff=6):
    - Features [0:18]   → Residue_1 raw features
    - Features [18:36]  → Residue_2 raw features
    - Features [36:54]  → Residue_3 raw features
    - Features [54:60]  → Residue_1 diff features
    - Features [60:66]  → Residue_2 diff features
    - Features [66:72]  → Residue_3 diff features
    
    **Key difference from interleaved structure:**
    - Block structure: [R1_raw, R2_raw, R3_raw, R1_diff, R2_diff, R3_diff]
    - Interleaved (OLD): [R1_raw, R1_diff, R2_raw, R2_diff, R3_raw, R3_diff]
    
    This is why we calculate indices as:
      raw_start = kept_position * cp_raw_feature_count
      diff_start = n_kept_residues * cp_raw_feature_count + kept_position * cp_diff_feature_count
    
    NOT as (which would be for interleaved):
      start = kept_position * (cp_raw_feature_count + cp_diff_feature_count)
    
    Residue Blocking:
    - Blocked residues have all their feature importances set to 0
    - For a blocked residue with cp_raw=18 and cp_diff=6, all 24 features are set to 0
    - Blocked residues are effectively removed from region importance calculations
    
    Aggregation Methods:
    1. 'sum-mean-topn': Original adaptive method
       - 1 residue: use 1
       - 2 residues: sum 2
       - 3-6 residues: top 3
       - ≥7 residues: top 5
    
    2. 'sum-mean-topn-mix4':
       - 1 residue: use 1
       - 2 residues: sum 2
       - 3 residues: top 3
       - ≥4 residues: top 4
    
    3. 'sum-mean-topn-mix5':
       - 1 residue: use 1
       - 2 residues: sum 2
       - 3-4 residues: top 3
       - ≥5 residues: top 4
    
    4. 'sum-mean-topn-mix6':
       - 1 residue: use 1
       - 2 residues: sum 2
       - 3-5 residues: top 3
       - ≥6 residues: top 4
    
    5. 'sum-mean-topn-filterimp-max3':
       - 1 residue: use 1
       - ≥2 residues: filter by max(mean, median), then select max 3 residues
       - Filters out low-importance residues before aggregation
    
    6. 'sum-mean-topn-filterimp-max4':
       - 1 residue: use 1
       - ≥2 residues: filter by max(mean, median), then select max 4 residues
       - Filters out low-importance residues before aggregation
    
    7. 'sum-mean-topn-mix6-filterimp-max4':
       - 1 residue: use 1
       - 2 residues: filter by max(mean, median), max 2 residues
       - 3-5 residues: filter by max(mean, median), max 3 residues
       - ≥6 residues: filter by max(mean, median), max 4 residues
       - Combines adaptive sizing with importance filtering
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> result = calculate_contact_region_rf_importance(
    ...     contact_regions, models, keep_residues, verbose=True
    ... )
    
    >>> # Using custom feature counts and selection
    >>> result = calculate_contact_region_rf_importance(
    ...     contact_regions, models, keep_residues,
    ...     cp_raw_feature_count=18,
    ...     cp_diff_feature_count=6,
    ...     use_topn_cp_raw_feature_count=10,
    ...     use_topn_cp_diff_feature_count=3,
    ...     verbose=True
    ... )
    
    >>> # Using different aggregation method with blocking
    >>> result = calculate_contact_region_rf_importance(
    ...     contact_regions, models, keep_residues,
    ...     aggregation_method='sum-mean-topn-mix4',
    ...     blocked_residues=[9, 839, 1106, 1107, 1108],  # 0-based
    ...     verbose=True
    ... )
    
    See Also
    --------
    ranking_consensus_contact_region : Higher-level function for complete analysis
    normalize_scores : Score normalization function
    """
    
    # 1. Extract feature importances from all models and compute mean
    n_residues = len(keep_residues)
    features_per_residue = cp_raw_feature_count + cp_diff_feature_count
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Calculating Contact Region RF Importance")
        print(f"{'='*80}")
        print(f"Feature configuration:")
        print(f"  Raw features per residue: {cp_raw_feature_count}")
        print(f"  Diff features per residue: {cp_diff_feature_count}")
        print(f"  Total features per residue: {features_per_residue}")
        
        # Display raw feature usage
        if use_topn_cp_raw_feature_count is None:
            raw_usage = "All"
        elif use_topn_cp_raw_feature_count == 0:
            raw_usage = "None (disabled)"
        else:
            raw_usage = f"Top {use_topn_cp_raw_feature_count}"
        print(f"  Using top-N raw features: {raw_usage}")
        
        # Display diff feature usage
        if use_topn_cp_diff_feature_count is None:
            diff_usage = "All"
        elif use_topn_cp_diff_feature_count == 0:
            diff_usage = "None (disabled)"
        else:
            diff_usage = f"Top {use_topn_cp_diff_feature_count}"
        print(f"  Using top-N diff features: {diff_usage}")
        
        print(f"Aggregation method: {aggregation_method}")
        print(f"Normalization method: {normalization_method}")
        print(f"Number of models in ensemble: {len(models)}")
        print(f"Total residues: {n_residues}")
        print(f"Kept residues: {np.sum(keep_residues)}")
        
        if blocked_residues is not None and len(blocked_residues) > 0:
            print(f"Blocked residues: {len(blocked_residues)} (0-based indices)")
    
    # Get indices of kept residues
    kept_residue_indices = np.where(keep_residues)[0]
    
    # Initialize storage for residue importances across all models
    all_residue_importances = []
    
    # Extract importances from each model
    for model_idx, model_dict in enumerate(models):
        model = model_dict['model']
        feature_importances = model.feature_importances_
        
        # Initialize residue importance array
        residue_importance = np.zeros(n_residues)
        
        # Calculate total number of kept residues for block indexing
        n_kept_residues = len(kept_residue_indices)
        
        # Process each kept residue
        for res_idx in kept_residue_indices:
            # CRITICAL: Find position in kept residues array (not original position!)
            kept_position = np.where(kept_residue_indices == res_idx)[0][0]
            
            # Check if this residue should be blocked
            if blocked_residues is not None and res_idx in blocked_residues:
                # Set importance to 0 for blocked residues
                residue_importance[res_idx] = 0.0
                continue
            
            # Calculate feature indices for this residue based on BLOCK structure
            # Features are organized as: [All Raw features | All Diff features]
            # Raw block: [Res1_raw, Res2_raw, ..., ResN_raw]
            # Diff block: [Res1_diff, Res2_diff, ..., ResN_diff]
            
            # Raw features for this residue (in the first block)
            raw_start_idx = kept_position * cp_raw_feature_count
            raw_end_idx = raw_start_idx + cp_raw_feature_count
            
            # Diff features for this residue (in the second block, after all raw features)
            diff_block_start = n_kept_residues * cp_raw_feature_count
            diff_start_idx = diff_block_start + (kept_position * cp_diff_feature_count)
            diff_end_idx = diff_start_idx + cp_diff_feature_count
            
            # Extract raw and diff feature importances
            raw_importances = feature_importances[raw_start_idx:raw_end_idx]
            diff_importances = feature_importances[diff_start_idx:diff_end_idx]
            
            # Select top-N features if specified
            if use_topn_cp_raw_feature_count is not None:
                if use_topn_cp_raw_feature_count == 0:
                    # 0 means don't use raw features at all
                    selected_raw_importance = 0.0
                else:
                    # Select top N raw features
                    n_select = min(use_topn_cp_raw_feature_count, len(raw_importances))
                    top_raw_indices = np.argsort(raw_importances)[-n_select:]
                    selected_raw_importance = np.sum(raw_importances[top_raw_indices])
            else:
                # None means use all raw features
                selected_raw_importance = np.sum(raw_importances)
            
            if use_topn_cp_diff_feature_count is not None:
                if use_topn_cp_diff_feature_count == 0:
                    # 0 means don't use diff features at all
                    selected_diff_importance = 0.0
                else:
                    # Select top N diff features
                    n_select = min(use_topn_cp_diff_feature_count, len(diff_importances))
                    top_diff_indices = np.argsort(diff_importances)[-n_select:]
                    selected_diff_importance = np.sum(diff_importances[top_diff_indices])
            else:
                # None means use all diff features
                selected_diff_importance = np.sum(diff_importances)
            
            # Sum raw and diff importances for this residue
            residue_importance[res_idx] = selected_raw_importance + selected_diff_importance
        
        all_residue_importances.append(residue_importance)
    
    # Convert to numpy array for easier manipulation
    all_residue_importances = np.array(all_residue_importances)  # Shape: (n_models, n_residues)
    
    # 2. Calculate mean and std across models for each residue
    mean_residue_importances = np.mean(all_residue_importances, axis=0)
    std_residue_importances = np.std(all_residue_importances, axis=0)
    
    # 3. Build residue importance dictionary
    residue_importance_dict = {}
    for res_idx in kept_residue_indices:
        residue_importance_dict[res_idx] = {
            'position': res_idx + 1,  # 1-based (for compatibility)
            'raw_score': mean_residue_importances[res_idx],  # Match original key name
            'model_scores': all_residue_importances[:, res_idx].tolist()  # Match original key name
        }
    
    # 4. Aggregate to region level using the specified aggregation method
    region_importance_dict = {}
    all_region_scores = []
    
    for region in contact_regions:
        region_id = region['contact_region_id']
        positions = region['positions']
        
        # Get valid positions (those in keep_residues and not blocked)
        valid_positions = [pos for pos in positions if keep_residues[pos]]
        
        # Filter out blocked residues if blocking is enabled
        if blocked_residues is not None:
            valid_positions = [pos for pos in valid_positions if pos not in blocked_residues]
        
        if len(valid_positions) > 0:
            # Get importance scores for all valid residues in this region
            region_residue_scores = [mean_residue_importances[pos] for pos in valid_positions]
            
            # Apply aggregation method
            region_score, top_indices, n_top_used = _aggregate_region_scores(
                region_residue_scores, aggregation_method
            )
            
            # Map top_indices back to original positions
            top_positions = [valid_positions[idx] for idx in top_indices]
            
            region_importance_dict[region_id] = {
                'region_id': region_id,
                'positions': positions,
                'size': region['size'],
                'residue_scores': region_residue_scores,
                'residue_positions': valid_positions,
                'region_score': region_score,
                'top_residues_idx': top_indices,
                'top_residues_pos': top_positions,
                'aggregation_method': aggregation_method,
                'n_top_used': n_top_used,
                'avg_correlation': region.get('avg_correlation', 0)
            }
            all_region_scores.append(region_score)
        else:
            # Empty region
            region_importance_dict[region_id] = {
                'region_id': region_id,
                'positions': positions,
                'size': region['size'],
                'residue_scores': [],
                'residue_positions': [],
                'region_score': 0,
                'top_residues_idx': [],
                'top_residues_pos': [],
                'aggregation_method': aggregation_method,
                'n_top_used': 0,
                'avg_correlation': region.get('avg_correlation', 0)
            }
            all_region_scores.append(0)
    
    # 5. Normalize scores
    all_region_scores = np.array(all_region_scores)
    normalized_scores = normalize_scores(all_region_scores, normalization_method, verbose=verbose)
    
    for i, region_id in enumerate(region_importance_dict):
        region_importance_dict[region_id]['normalized_score'] = normalized_scores[i]
    
    # 6. Build return dictionary
    importance_dict = {
        'residue_importance': residue_importance_dict,
        'region_importance': region_importance_dict,
        'normalization_method': normalization_method,
        'aggregation_method': aggregation_method
    }
    
    # 7. Print summary statistics if verbose
    if verbose:
        print(f"\nContact Region RF Importance Analysis Summary:")
        print(f"  Aggregation method: {aggregation_method}")
        print(f"  Normalization method: {normalization_method}")
        print(f"  Residues analyzed: {len(residue_importance_dict)}")
        print(f"  Contact regions analyzed: {len(region_importance_dict)}")
            
    return importance_dict


def _aggregate_region_scores(residue_scores, aggregation_method):
    """
    Aggregate residue scores within a region based on specified method.
    
    Parameters
    ----------
    residue_scores : list of float
        List of importance scores for residues in a region.
    aggregation_method : str
        Aggregation method to use.
    
    Returns
    -------
    region_score : float
        Aggregated score for the region.
    top_indices : list of int
        Indices of residues used for aggregation.
    n_top : int
        Number of top residues used.
    """
    scores_array = np.array(residue_scores)
    n_residues = len(scores_array)
    
    if n_residues == 0:
        return 0, [], 0
    
    if n_residues == 1:
        return scores_array[0], [0], 1
    
    # Determine aggregation strategy based on method and region size
    if 'filterimp' in aggregation_method:
        # Filter-based methods
        return _aggregate_with_filtering(scores_array, aggregation_method)
    else:
        # Standard top-N methods
        return _aggregate_topn(scores_array, aggregation_method)


def _aggregate_with_filtering(scores_array, aggregation_method):
    """
    Aggregate scores with importance filtering.
    
    Parameters
    ----------
    scores_array : numpy.ndarray
        Array of residue importance scores.
    aggregation_method : str
        Aggregation method specifying filtering strategy.
    
    Returns
    -------
    region_score : float
        Aggregated score for the region.
    top_indices : list of int
        Indices of residues used for aggregation.
    n_top : int
        Number of top residues used.
    """
    n_residues = len(scores_array)
    
    # Calculate filtering threshold
    mean_score = np.mean(scores_array)
    median_score = np.median(scores_array)
    threshold = max(mean_score, median_score)
    
    # Filter residues above threshold
    filtered_indices = np.where(scores_array >= threshold)[0]
    
    if len(filtered_indices) == 0:
        # If no residues pass filter, use the top residue
        top_idx = np.argmax(scores_array)
        return scores_array[top_idx], [top_idx], 1
    
    # Determine max residues based on method
    if aggregation_method == 'sum-mean-topn-filterimp-max3':
        max_residues = 3
    elif aggregation_method == 'sum-mean-topn-filterimp-max4':
        max_residues = 4
    elif aggregation_method == 'sum-mean-topn-mix6-filterimp-max4':
        # Adaptive based on region size
        if n_residues == 2:
            max_residues = 2
        elif n_residues <= 5:
            max_residues = 3
        else:
            max_residues = 4
    else:
        max_residues = min(5, n_residues)
    
    # Select top residues from filtered set
    filtered_scores = scores_array[filtered_indices]
    if len(filtered_indices) <= max_residues:
        # Use all filtered residues
        selected_indices = filtered_indices
    else:
        # Select top max_residues from filtered set
        top_in_filtered = np.argsort(filtered_scores)[-max_residues:]
        selected_indices = filtered_indices[top_in_filtered]
    
    region_score = np.sum(scores_array[selected_indices])
    return region_score, selected_indices.tolist(), len(selected_indices)


def _aggregate_topn(scores_array, aggregation_method):
    """
    Aggregate scores using top-N strategy.
    
    Parameters
    ----------
    scores_array : numpy.ndarray
        Array of residue importance scores.
    aggregation_method : str
        Aggregation method specifying top-N strategy.
    
    Returns
    -------
    region_score : float
        Aggregated score for the region.
    top_indices : list of int
        Indices of residues used for aggregation.
    n_top : int
        Number of top residues used.
    """
    n_residues = len(scores_array)
    
    # Determine n_top based on aggregation method
    if aggregation_method == 'sum-mean-topn-mix4':
        # Mix4: 1→1, 2→2, 3→3, ≥4→4
        if n_residues == 2:
            n_top = 2
        elif n_residues == 3:
            n_top = 3
        else:  # ≥4
            n_top = 4
    
    elif aggregation_method == 'sum-mean-topn-mix5':
        # Mix5: 1→1, 2→2, 3-4→3, ≥5→4
        if n_residues == 2:
            n_top = 2
        elif n_residues <= 4:  # 3-4
            n_top = 3
        else:  # ≥5
            n_top = 4
    
    elif aggregation_method == 'sum-mean-topn-mix6':
        # Mix6: 1→1, 2→2, 3-5→3, ≥6→4
        if n_residues == 2:
            n_top = 2
        elif n_residues <= 5:  # 3-5
            n_top = 3
        else:  # ≥6
            n_top = 4
    
    else:
        # Default to sum-mean-topn behavior
        if n_residues == 2:
            n_top = 2
        elif 3 <= n_residues <= 6:
            n_top = 3
        else:  # n_residues >= 7
            n_top = 5
    
    # Ensure n_top doesn't exceed available residues
    n_top = min(n_top, n_residues)
    
    # Get indices of top N residues
    top_indices = np.argsort(scores_array)[-n_top:].tolist()
    
    # Calculate region score as sum of top N residues
    region_score = np.sum(scores_array[top_indices])
    
    return region_score, top_indices, n_top


# ===================== normalization method =====================
def normalize_scores(scores, method='robust_minmax', power=4, verbose=True):
    """
    Normalize importance scores to 0-1 range using various methods.
    
    This function applies different normalization strategies to transform raw
    importance scores into a standardized 0-1 range, making scores comparable
    across different analyses and facilitating interpretation.
    
    Parameters
    ----------
    scores : numpy.ndarray
        Array of raw importance scores to be normalized.
    method : str, default='robust_minmax'
        Normalization method to apply. Options:
        - 'robust_minmax': Min-max normalization using 5th/95th percentiles
          to reduce sensitivity to outliers
        - 'percentile': Rank-based percentile transformation (rank/n)
        - 'zscore': Z-score standardization followed by sigmoid mapping
        - 'log_transform': Log transformation followed by min-max scaling
        - 'power_zscore': Power transformation, then z-score, then sigmoid
        - 'minmax': Simple min-max normalization (default fallback)
    power : float, default=4
        Power for 'power_zscore' method. Higher values increase separation
        between high and low scores. Ignored for other methods.
    verbose : bool, default=True
        If True, prints normalization method and input score statistics.
        If False, suppresses all console output.
    
    Returns
    -------
    normalized_scores : numpy.ndarray
        Normalized scores in 0-1 range, same shape as input.
        Returns input unchanged if all scores are 0 or array is empty.
    
    Notes
    -----
    Method Details:
    
    1. robust_minmax:
       - Clips to 5th-95th percentile range
       - Applies min-max: (x - p5) / (p95 - p5)
       - Robust to outliers
    
    2. percentile:
       - Ranks scores and divides by n
       - Maintains ordinal relationships
       - Uniform distribution in output
    
    3. zscore:
       - Standardizes: z = (x - mean) / std
       - Maps to 0-1 via sigmoid: 1 / (1 + exp(-z/2))
       - Preserves relative differences
    
    4. log_transform:
       - Applies: log(x + epsilon)
       - Then min-max normalization
       - Compresses high values
    
    5. power_zscore:
       - Transforms: x^power
       - Then z-score standardization
       - Then sigmoid mapping
       - Emphasizes differences in high scores
    
    6. minmax (default):
       - Simple: (x - min) / (max - min)
       - Sensitive to outliers
    
    Examples
    --------
    >>> # Robust normalization for scores with outliers
    >>> scores = np.array([0.001, 0.002, 0.003, 0.100])  # One outlier
    >>> norm_scores = normalize_scores(scores, method='robust_minmax')
    
    >>> # Emphasize high-importance differences
    >>> norm_scores = normalize_scores(scores, method='power_zscore', power=4)
    
    >>> # Silent mode
    >>> norm_scores = normalize_scores(scores, method='zscore', verbose=False)
    
    See Also
    --------
    calculate_contact_region_rf_importance : Uses this for score normalization
    """
    if len(scores) == 0 or np.max(scores) == 0:
        return scores

    # Print debug info if verbose
    if verbose:
        print(f"\n[DEBUG] Normalizing with method: {method}")
        print(f"[DEBUG] Input scores - Min: {np.min(scores):.6f}, Max: {np.max(scores):.6f}, Mean: {np.mean(scores):.6f}")
    
    if method == 'robust_minmax':
        # Use 5th and 95th percentiles instead of min and max
        p5 = np.percentile(scores, 5)
        p95 = np.percentile(scores, 95)
        
        if p95 > p5:
            # Clip to [p5, p95] range
            clipped_scores = np.clip(scores, p5, p95)
            normalized_scores = (clipped_scores - p5) / (p95 - p5)
        else:
            normalized_scores = np.ones_like(scores) * 0.5
            
    elif method == 'percentile':
        # Use percentile rank as normalized value
        normalized_scores = stats.rankdata(scores, method='average') / len(scores)
        
    elif method == 'zscore':
        # Z-score standardization, then sigmoid mapping to 0-1
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score > 0:
            z_scores = (scores - mean_score) / std_score
            # Use sigmoid function to map to 0-1
            normalized_scores = 1 / (1 + np.exp(-z_scores/2))
        else:
            normalized_scores = np.ones_like(scores) * 0.5
            
    elif method == 'log_transform':
        # Log transform followed by normalization
        epsilon = 1e-10
        log_scores = np.log(scores + epsilon)
        min_log = np.min(log_scores)
        max_log = np.max(log_scores)
        
        if max_log > min_log:
            normalized_scores = (log_scores - min_log) / (max_log - min_log)
        else:
            normalized_scores = np.zeros_like(scores)

    elif method == 'power_zscore':
        # Step 1: Power transform
        if power == 1.0:
            # When power=1, skip transform and use original scores
            transformed_scores = scores.copy()
        else:            
            # Apply power transformation
            transformed_scores = np.power(scores, power)
        
        # Step 2: Z-score standardization
        mean_transformed = np.mean(transformed_scores)
        std_transformed = np.std(transformed_scores)
        
        if std_transformed > 0:
            z_scores = (transformed_scores - mean_transformed) / std_transformed
            
            # Step 3: Sigmoid mapping
            normalized_scores = 1 / (1 + np.exp(-z_scores/2))
        else:
            normalized_scores = np.ones_like(scores) * 0.5            
            
    else:  # default: simple minmax
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)

    return normalized_scores
