# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from scipy import stats

from joblib import Parallel, delayed
import multiprocessing


# =========================== contact region importance ===========================
def ranking_consensus_contact_region(contact_regions, models, keep_residues,
                                    protein_sequence=None,
                                    normalization_method='power_zscore',
                                    aggregation_method='sum-mean-topn',
                                    output_tsv=None,
                                    back_res_dict=False,
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
        'sum-mean-topn': Adaptive top-N selection based on region size
    output_tsv : str, optional
        Path for TSV output file. If None, no file is saved. Default is None.
    back_res_dict : bool, default=False
        If True, returns tuple (df, dict). If False, returns only df.
        The dict contains complete contact_region_rf_importance results.
    verbose : bool, default=True
        If True, prints detailed progress and ranking information.
        If False, suppresses all console output.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with CCR ranking results containing columns:
        - CCR_rank: Rank by importance (1-based, int)
        - CCR_ID: Contact region identifier (1-based, int)
        - CCR_size: Number of residues (int)
        - CCR_range: Position range string (1-based)
        - raw_score: Raw RF importance score (float)
        - norm_score: Normalized importance score 0-1 (float)
        - CCR_all_residues: Complete residue annotation string
    contact_region_rf_importance : dict, optional
        Only returned if back_res_dict=True. Contains:
        - 'residue_importance': Dict of per-residue RF scores
        - 'region_importance': Dict of per-region RF scores and metadata
        - 'normalization_method': Normalization method used
        - 'aggregation_method': Aggregation method used
    
    Notes
    -----
    Output Format (all 1-based indexing):
    - DataFrame and TSV have identical content
    - CCR_rank: Rank by importance (1 = highest)
    - CCR_ID: Contact region identifier (converted to 1-based)
    - CCR_size: Number of residues in region
    - CCR_range: Position range in format "start-end" (1-based)
    - raw_score: Raw RF importance score
    - norm_score: Normalized importance score (0-1 range)
    - CCR_all_residues: Complete residue annotation
      Format: "123A,124T,125G" (position + amino acid letter)
      Includes ALL positions in range, even if not in kept_residues
    
    Return Behavior:
    - If back_res_dict=False: returns df
    - If back_res_dict=True: returns (df, contact_region_rf_importance)
    
    Examples
    --------
    >>> # Example 1: Get DataFrame only
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence="MDEFGHIK...",
    ...     verbose=True
    ... )
    >>> print(df.head())
    
    >>> # Example 2: Export to TSV and get DataFrame
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cas9_sequence,
    ...     output_tsv="cas9_ccr_ranking.tsv",
    ...     verbose=True
    ... )
    >>> print(f"Exported {len(df)} regions")
    
    >>> # Example 3: Get both DataFrame and dict
    >>> df, result_dict = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cas9_sequence,
    ...     output_tsv="cas9_ccr_ranking.tsv",
    ...     back_res_dict=True,
    ...     verbose=True
    ... )
    >>> print(f"Top region score: {result_dict['region_importance'][0]['region_score']:.4f}")
    
    >>> # Example 4: Silent mode with DataFrame
    >>> df = ranking_consensus_contact_region(
    ...     contact_regions, models, keep_residues,
    ...     protein_sequence=cpf1_sequence,
    ...     output_tsv="cpf1_ccr_ranking.tsv",
    ...     verbose=False
    ... )
    
    See Also
    --------
    calculate_contact_region_rf_importance : Core computation function
    normalize_scores : Score normalization methods
    _export_ccr_ranking_to_tsv : TSV export helper
    _create_ccr_ranking_dataframe : DataFrame creation helper
    """
    
    # Step 1: Calculate contact region RF importance
    contact_region_rf_importance = calculate_contact_region_rf_importance(
        contact_regions=contact_regions,
        models=models,
        keep_residues=keep_residues,
        normalization_method=normalization_method,
        aggregation_method=aggregation_method,
        verbose=verbose
    )
    
    # Step 2: Extract and sort regions
    cr_importance = contact_region_rf_importance['region_importance']
    sorted_crs = sorted(cr_importance.items(), 
                       key=lambda x: x[1]['region_score'], 
                       reverse=True)
    
    # Step 3: Print ranking table if verbose
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
    
    # Step 4: Create DataFrame
    df = _create_ccr_ranking_dataframe(
        sorted_crs=sorted_crs,
        protein_sequence=protein_sequence
    )
    
    # Step 5: Export to TSV if requested
    if output_tsv is not None:
        df.to_csv(output_tsv, sep='\t', index=False)
        if verbose:
            print(f"✓ CCR ranking exported to: {output_tsv}")
            print(f"  Total regions: {len(df)}")
            print(f"  Format: TSV with {len(df.columns)} columns (1-based indexing)\n")
    
    # Step 6: Return based on back_res_dict flag
    if back_res_dict:
        return df, contact_region_rf_importance
    else:
        return df


def _create_ccr_ranking_dataframe(sorted_crs, protein_sequence):
    """
    Create pandas DataFrame from sorted CCR ranking results.
    
    Helper function to convert ranked contact region results into a structured
    pandas DataFrame with consistent formatting.
    
    Parameters
    ----------
    sorted_crs : list of tuple
        Sorted list of (region_id, region_info) tuples from ranking.
    protein_sequence : str or None
        Complete protein amino acid sequence (single-letter code).
        If None, residue annotations will use 'X' as placeholder.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns (all 1-based indexing):
        - CCR_rank (int): Rank by RF importance
        - CCR_ID (int): Contact region identifier
        - CCR_size (int): Number of residues
        - CCR_range (str): Position range "start-end"
        - raw_score (float): Raw RF importance score
        - norm_score (float): Normalized score (0-1)
        - CCR_all_residues (str): Complete residue list with amino acids
    
    Notes
    -----
    For CCR_all_residues column:
    - Format: "pos1AA1,pos2AA2,pos3AA3,..."
    - Example: "1000M,1001T,1002G,1003R,1004K"
    - Includes ALL positions in CCR range
    - If protein_sequence is None, uses 'X' for amino acids
    
    Examples
    --------
    >>> df = _create_ccr_ranking_dataframe(sorted_regions, cas9_sequence)
    >>> print(df.dtypes)
    CCR_rank            int64
    CCR_ID              int64
    CCR_size            int64
    CCR_range          object
    raw_score         float64
    norm_score        float64
    CCR_all_residues   object
    dtype: object
    """
    data_rows = []
    
    for rank, (region_id, region_info) in enumerate(sorted_crs, 1):
        positions = region_info['positions']
        
        # Calculate CCR range (1-based)
        if len(positions) == 0:
            ccr_range = "N/A"
            ccr_all_residues = ""
        elif len(positions) == 1:
            ccr_range = str(positions[0])
            ccr_all_residues = _format_residue_annotation(
                positions[0], positions[0], protein_sequence
            )
        else:
            min_pos = min(positions)
            max_pos = max(positions)
            ccr_range = f"{min_pos}-{max_pos}"
            ccr_all_residues = _format_residue_annotation(
                min_pos, max_pos, protein_sequence
            )
        
        data_rows.append({
            'CCR_rank': rank,
            'CCR_ID': region_info['region_id'] + 1,
            'CCR_size': region_info['size'],
            'CCR_range': ccr_range,
            'raw_score': region_info['region_score'],
            'norm_score': region_info['normalized_score'],
            'CCR_all_residues': ccr_all_residues
        })
    
    df = pd.DataFrame(data_rows)
    
    return df


def _export_ccr_ranking_to_tsv(sorted_crs, protein_sequence, output_path, verbose=True):
    """
    Export ranked consensus contact regions to TSV file.
    
    Helper function to format and write CCR ranking results to a tab-separated file
    with comprehensive residue annotations.
    
    Parameters
    ----------
    sorted_crs : list of tuple
        Sorted list of (region_id, region_info) tuples from ranking.
    protein_sequence : str or None
        Complete protein amino acid sequence (single-letter code).
        If None, residue annotations will only include positions.
    output_path : str
        Output file path for TSV file.
    verbose : bool, default=True
        If True, prints export confirmation message.
    
    Notes
    -----
    Creates TSV with the following columns (all 1-based indexing):
    - CCR_rank: Rank by RF importance
    - CCR_ID: Contact region identifier
    - CCR_size: Number of residues
    - CCR_range: Position range "start-end"
    - raw_score: Raw RF importance score
    - norm_score: Normalized score (0-1)
    - CCR_all_residues: Complete residue list with amino acid letters
    
    For CCR_all_residues column:
    - Format: "pos1AA1,pos2AA2,pos3AA3,..."
    - Example: "1000M,1001T,1002G,1003R,1004K"
    - Includes ALL positions in CCR range, regardless of kept_residues
    - If protein_sequence is None, format becomes: "pos1X,pos2X,..."
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow([
            'CCR_rank',
            'CCR_ID', 
            'CCR_size',
            'CCR_range',
            'raw_score',
            'norm_score',
            'CCR_all_residues'
        ])
        
        # Write data rows
        for rank, (region_id, region_info) in enumerate(sorted_crs, 1):
            positions = region_info['positions']
            
            # Calculate CCR range (1-based)
            if len(positions) == 0:
                ccr_range = "N/A"
                ccr_all_residues = ""
            elif len(positions) == 1:
                ccr_range = str(positions[0])
                ccr_all_residues = _format_residue_annotation(
                    positions[0], positions[0], protein_sequence
                )
            else:
                min_pos = min(positions)
                max_pos = max(positions)
                ccr_range = f"{min_pos}-{max_pos}"
                ccr_all_residues = _format_residue_annotation(
                    min_pos, max_pos, protein_sequence
                )
            
            writer.writerow([
                rank,                                      # CCR_rank (1-based)
                region_info['region_id'] + 1,             # CCR_ID (convert to 1-based)
                region_info['size'],                       # CCR_size
                ccr_range,                                 # CCR_range (1-based)
                f"{region_info['region_score']:.6f}",     # raw_score
                f"{region_info['normalized_score']:.6f}", # norm_score
                ccr_all_residues                           # CCR_all_residues
            ])
    
    if verbose:
        print(f"✓ CCR ranking exported to: {output_path}")
        print(f"  Total regions: {len(sorted_crs)}")
        print(f"  Format: TSV with 7 columns (1-based indexing)\n")


def _format_residue_annotation(start_pos, end_pos, protein_sequence):
    """
    Format residue range as annotated string with amino acid letters.
    
    Parameters
    ----------
    start_pos : int
        Start position (1-based).
    end_pos : int
        End position (1-based, inclusive).
    protein_sequence : str or None
        Complete protein sequence. If None, uses 'X' as placeholder.
    
    Returns
    -------
    annotation : str
        Formatted string: "pos1AA1,pos2AA2,..."
        Example: "1000M,1001T,1002G"
    
    Notes
    -----
    - Includes ALL positions from start_pos to end_pos (inclusive)
    - If protein_sequence is None, all amino acids are 'X'
    - If position exceeds sequence length, amino acid is 'X'
    """
    residues = []
    for pos in range(start_pos, end_pos + 1):
        if protein_sequence is None:
            aa = 'X'
        elif pos - 1 < len(protein_sequence):
            aa = protein_sequence[pos - 1]  # Convert 1-based to 0-based
        else:
            aa = 'X'
        
        residues.append(f"{pos}{aa}")
    
    return ",".join(residues)


def calculate_contact_region_rf_importance(contact_regions, models, keep_residues, 
                                         normalization_method='power_zscore',
                                         aggregation_method='sum-mean-topn',
                                         verbose=True):
    """
    Calculate Random Forest importance scores for consensus contact regions.
    
    This function computes RF importance for each contact region by aggregating
    the feature importance scores of residues within each region. Each region is
    scored independently without merging.
    
    Parameters
    ----------
    contact_regions : list of dict
        List of contact region dictionaries. Each dict must contain:
        - 'contact_region_id': Unique region identifier (0-based)
        - 'positions': Array/list of residue positions (0-based) in the region
        - 'size': Number of residues in the region (optional)
        - 'type': Region type descriptor (optional)
        - 'avg_correlation': Average internal correlation (optional)
    models : list of dict
        List of trained Random Forest model dictionaries. Each dict must contain:
        - 'model': sklearn.ensemble.RandomForestClassifier instance
    keep_residues : numpy.ndarray
        Boolean array of length n_residues indicating which residues were
        retained for RF training (True = retained, False = filtered).
    normalization_method : str, default='power_zscore'
        Method for normalizing region scores to 0-1 range. Options:
        - 'power_zscore': Apply power transform, then z-score, then sigmoid
        - 'robust_minmax': Min-max using 5th and 95th percentiles
        - 'percentile': Rank-based percentile transformation
        - 'zscore': Standard z-score with sigmoid mapping
        - 'log_transform': Log transformation followed by min-max
        - 'minmax': Simple min-max normalization
    aggregation_method : str, default='sum-mean-topn'
        Method for aggregating residue RF scores within each region.
        Currently only 'sum-mean-topn' is supported:
        - Size 1: Use single residue score
        - Size 2: Sum of both residues
        - Size 3-6: Sum of top 3 residues
        - Size ≥7: Sum of top 5 residues
    verbose : bool, default=True
        If True, prints detailed progress information and top 10 regions.
        If False, suppresses all console output.
    
    Returns
    -------
    importance_dict : dict
        Dictionary containing comprehensive importance results:
        - 'residue_importance': Dict mapping residue position (0-based) to:
          - 'position': 1-based position
          - 'raw_score': Raw RF importance (mean across models)
          - 'model_scores': List of scores from each model
        - 'region_importance': Dict mapping region_id (0-based) to:
          - 'region_id': Original contact_region_id
          - 'positions': List of 1-based positions
          - 'size': Number of residues
          - 'type': Region type
          - 'residue_scores': List of raw residue scores
          - 'residue_positions': List of 1-based positions
          - 'region_score': Aggregated RF importance score
          - 'top_residues_idx': Indices of top residues used
          - 'top_residues_pos': 1-based positions of top residues
          - 'aggregation_method': Method used
          - 'n_top_used': Number of residues used in aggregation
          - 'normalized_score': Normalized score (0-1)
          - 'avg_correlation': Average internal correlation
        - 'normalization_method': Method used for normalization
        - 'aggregation_method': Method used for aggregation
    
    Notes
    -----
    Feature Importance Calculation:
    - Each residue has 24 features: 18 raw + 6 diff
    - Feature importance is summed across all 24 features per residue
    - Final residue score is the mean across all RF models
    
    Region Score Aggregation:
    - Adaptive top-N selection based on region size
    - Prevents bias from very large regions
    - Maintains sensitivity for small functional units
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> result = calculate_contact_region_rf_importance(
    ...     contact_regions, models, keep_residues, verbose=True
    ... )
    >>> top_region = result['region_importance'][0]
    >>> print(f"Top region score: {top_region['region_score']:.4f}")
    
    >>> # Silent mode with custom normalization
    >>> result = calculate_contact_region_rf_importance(
    ...     contact_regions, models, keep_residues,
    ...     normalization_method='robust_minmax',
    ...     verbose=False
    ... )
    
    See Also
    --------
    normalize_scores : Score normalization function
    calculate_residue_rf_importance : Residue-level importance calculation
    """
    
    if verbose:
        print(f"\nCalculating contact region RF importance using {aggregation_method} method...")
    
    # 1. Collect feature importances from all models
    all_feature_importances = []
    for model_dict in models:
        rf_model = model_dict['model']
        all_feature_importances.append(rf_model.feature_importances_)
    
    # Feature dimension info
    n_kept_residues = np.sum(keep_residues)
    n_features_per_residue = 24  # 18 raw + 6 diff
    
    # 2. Calculate residue-level importance
    kept_indices = np.where(keep_residues)[0]
    residue_importance_dict = {}
    
    for pos in kept_indices:
        # Find position in kept residues array
        kept_position = np.where(kept_indices == pos)[0][0]
        
        # Collect feature sums from 5 models
        model_sums = []
        
        for feature_imp in all_feature_importances:
            # Extract 24 features for this residue
            start_idx = kept_position * n_features_per_residue
            end_idx = start_idx + n_features_per_residue
            residue_features = feature_imp[start_idx:end_idx]
            
            # Sum features
            feature_sum = np.sum(residue_features)
            model_sums.append(feature_sum)
        
        # Sum-Mean: Average across 5 models
        residue_score = np.mean(model_sums)
        
        residue_importance_dict[pos] = {
            'position': pos + 1,  # 1-based
            'raw_score': residue_score,
            'model_scores': model_sums
        }
    
    # 3. Define aggregation function
    def aggregate_scores(residue_scores, method=aggregation_method):
        """
        Aggregate residue scores within a contact region.
        
        Parameters
        ----------
        residue_scores : list or np.ndarray
            RF importance scores for residues in the region.
        method : str
            Aggregation method (currently only 'sum-mean-topn').
        
        Returns
        -------
        score : float
            Aggregated region score.
        top_indices : list
            Indices of residues used in aggregation.
        """
        if not residue_scores:
            return 0, []
            
        scores_array = np.array(residue_scores)
        n_residues = len(scores_array)
        
        if method == 'sum-mean-topn':
            sorted_scores = np.sort(scores_array)
            
            if n_residues == 1:
                score = sorted_scores[-1]
                top_indices = [0]
                top_n = 1
            elif n_residues == 2:
                score = np.sum(sorted_scores[-2:])
                top_indices = np.argsort(scores_array)[-2:]
                top_n = 2
            elif 3 <= n_residues <= 6:
                top_n = min(3, n_residues)
                score = np.sum(sorted_scores[-top_n:])
                top_indices = np.argsort(scores_array)[-top_n:]
            else:  # n_residues >= 7
                top_n = min(5, n_residues)
                score = np.sum(sorted_scores[-top_n:])
                top_indices = np.argsort(scores_array)[-top_n:]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
        return score, top_indices
    
    # 4. Calculate importance for each contact region
    region_importance_dict = {}
    all_region_scores = []
    
    for region_id, region in enumerate(contact_regions):
        # Get region position info
        if 'positions' in region:
            region_positions = region['positions']
        else:
            # If no positions field, compute from indices
            region_positions = kept_indices[region['indices']]
        
        # Collect residue scores for this region
        residue_scores = []
        residue_positions = []
        
        for pos in region_positions:
            if pos in residue_importance_dict:
                residue_scores.append(residue_importance_dict[pos]['raw_score'])
                residue_positions.append(pos + 1)  # 1-based
        
        # Aggregate using specified method
        if residue_scores:
            region_score, top_residues = aggregate_scores(residue_scores, aggregation_method)
            
            region_importance_dict[region_id] = {
                'region_id': region['contact_region_id'],
                'positions': region_positions + 1 if isinstance(region_positions, np.ndarray) else [p + 1 for p in region_positions],
                'size': len(region_positions),
                'type': region.get('type', 'contact_region'),
                'residue_scores': residue_scores,
                'residue_positions': residue_positions,
                'region_score': region_score,
                'top_residues_idx': top_residues,
                'top_residues_pos': [residue_positions[i] for i in top_residues],
                'aggregation_method': aggregation_method,
                'n_top_used': len(top_residues),
                'avg_correlation': region.get('avg_correlation', 0)
            }
            all_region_scores.append(region_score)
        else:
            region_importance_dict[region_id] = {
                'region_id': region['contact_region_id'],
                'positions': [],
                'size': 0,
                'type': region.get('type', 'contact_region'),
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