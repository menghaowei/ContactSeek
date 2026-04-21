# -*- coding: UTF-8 -*-

import numpy as np

# last edition: 2026-02-16

####################################################################################
# function define [calculation part]
####################################################################################
def top_n_average(arr: np.ndarray, n: int) -> float:
    """
    Calculate the average of the top n largest values in an array.
    
    Args:
        arr: Input numpy array of any shape.
        n: Number of top values to average.
    
    Returns:
        The arithmetic mean of the n largest values.
    
    Example:
        >>> arr = np.array([[1, 5], [3, 9]])
        >>> top_n_average(arr, 2)
        7.0
    """
    top_n = np.sort(arr.flatten())[-n:]
    return np.mean(top_n)


def get_top_n(arr: np.ndarray, n: int) -> np.array:
    """
    Extract the top n largest values from an array in descending order.
    
    Args:
        arr: Input numpy array of any shape.
        n: Number of top values to extract.
    
    Returns:
        A 1D array containing the n largest values sorted in descending order.
    
    Example:
        >>> arr = np.array([[1, 5], [3, 9]])
        >>> get_top_n(arr, 3)
        array([9, 5, 3])
    """
    return np.sort(arr.flatten())[-n:][::-1]


def round_cp_matrix(input_np_mat, np_limit=1e-5, np_round_num=4):
    """
    Round matrix values and set small positive values to zero.
    
    Sets all values below the specified threshold to zero, then rounds
    the remaining values to a specified number of decimal places. This is
    useful for cleaning up numerical artifacts in computation results.
    
    Args:
        input_np_mat: Input numpy matrix to be processed.
        np_limit: Threshold below which values are set to zero. Default is 1e-5.
        np_round_num: Number of decimal places for rounding. Default is 4.
    
    Returns:
        A numpy array with small values zeroed and remaining values rounded.
    
    Note:
        This function modifies the input array in-place before rounding.
    
    Example:
        >>> mat = np.array([[0.00001, 0.12345], [0.00000001, 1.56789]])
        >>> round_cp_matrix(mat, np_limit=1e-5, np_round_num=2)
        array([[0.  , 0.12],
               [0.  , 1.57]])
    """
    input_np_mat[input_np_mat < np_limit] = 0
    return np.round(input_np_mat, np_round_num)


def round_abs_cp_matrix(input_np_mat, np_abs_limit=1e-5, np_round_num=4):
    """
    Round matrix values and set values with small absolute magnitude to zero.
    
    Sets all values whose absolute value is below the specified threshold to zero,
    then rounds the remaining values to a specified number of decimal places. Unlike
    round_cp_matrix, this function handles both positive and negative small values.
    
    Args:
        input_np_mat: Input numpy matrix or array-like object to be processed.
        np_abs_limit: Absolute value threshold below which values are set to zero. 
                      Default is 1e-5.
        np_round_num: Number of decimal places for rounding. Default is 4.
    
    Returns:
        A numpy array with small-magnitude values zeroed and remaining values rounded.
    
    Note:
        Creates a copy of the input as a numpy array, so the original input is not modified.
    
    Example:
        >>> mat = np.array([[-0.00001, 0.12345], [0.00000001, -1.56789]])
        >>> round_abs_cp_matrix(mat, np_abs_limit=1e-5, np_round_num=2)
        array([[ 0.  ,  0.12],
               [ 0.  , -1.57]])
    """
    input_np_mat = np.array(input_np_mat)
    input_np_mat[np.abs(input_np_mat) < np_abs_limit] = 0
    return np.round(input_np_mat, np_round_num)


####################################################################################
# function define [top-n extraction - REFACTORED VERSION]
####################################################################################
def query_cp_with_top_n(
    input_array: np.ndarray,
    from_resi_idx_list: list,
    query_idx_list: list,
    top_n: int = 3,
    ref_array: np.ndarray = None
) -> dict:
    """
    Extract and compare contact probability between residues and query regions.
    
    Analyzes contact probability matrix to extract and compare interactions between
    specified residues (from_resi_idx_list) and specified nucleic acid/target regions 
    (query_idx_list). For each residue-query pair, calculates maximum contact probability, 
    top-n average, and top-n individual values.
    
    Args:
        input_array: Contact probability matrix for the input condition.
                     Shape: (N, N) where N is the total number of residues/positions.
                     
        from_resi_idx_list: List of 1-based residue indices to analyze. 
                           Length: n (number of residues to query).
                           Example: [1, 5, 10, 100] queries residues at these positions.
                           
        query_idx_list: List of tuples specifying target query regions in Python-style
                        indexing (start, end). Length: m (number of query regions).
                        Example: [(1368, 1391), (1400, 1420)] for two query regions.
                        
        top_n: Number of top contact values to extract for averaging and analysis. 
               Default is 3.
               
        ref_array: Optional reference contact probability matrix for comparison.
                   If None, all diff values will be zero. If provided, must have
                   the same shape as input_array. Default is None.
    
    Returns:
        A dictionary containing contact probability analyses with the following keys:
            - "ref_prob": Maximum contact probabilities for each residue-query pair (reference).
                         Shape: (n, m) where n = len(from_resi_idx_list), m = len(query_idx_list)
                         
            - "input_prob": Maximum contact probabilities for each residue-query pair (input).
                           Shape: (n, m)
                           
            - "diff_prob": Differences (input - ref) in maximum contact probabilities.
                          Shape: (n, m). All zeros if ref_array is None.
                          
            - "mean_top_ref_prob": Top-n averaged contact probabilities (reference).
                                  Shape: (n, m)
                                  
            - "mean_top_input_prob": Top-n averaged contact probabilities (input).
                                    Shape: (n, m)
                                    
            - "mean_top_diff_prob": Differences (input - ref) in top-n averaged probabilities.
                                   Shape: (n, m). All zeros if ref_array is None.
                                   
            - "top_n_ref_prob": Individual top-n contact values (reference).
                               Shape: (n, top_n * m). For each residue, contains 
                               top_n values for each of m query regions concatenated.
                               
            - "top_n_input_prob": Individual top-n contact values (input).
                                 Shape: (n, top_n * m)
                                 
            - "top_n_diff_prob": Differences (input - ref) in top-n values.
                                Shape: (n, top_n * m). All zeros if ref_array is None.
    
    Raises:
        ValueError: If input_array and ref_array have different shapes (when ref_array is not None).
        ValueError: If from_resi_idx_list contains indices outside valid range.
    
    Note:
        All probability values are rounded to 4 decimal places. The function calculates
        three types of metrics for each residue-query pair: (1) maximum value, 
        (2) average of top-n values, and (3) individual top-n values as arrays.
        
        The 1-based indices in from_resi_idx_list are converted to 0-based for array indexing.
    
    Example:
        >>> input_mat = np.random.rand(1500, 1500)
        >>> ref_mat = np.random.rand(1500, 1500)
        >>> from_resi = [1, 10, 100, 500]  # 4 residues
        >>> query_regions = [(1368, 1391), (1400, 1420)]  # 2 query regions
        >>> result = query_cp_with_top_n(input_mat, from_resi, query_regions, top_n=3, ref_array=ref_mat)
        >>> result["input_prob"].shape
        (4, 2)
        >>> result["top_n_input_prob"].shape
        (4, 6)  # 4 residues × (3 top values × 2 query regions)
    """
    
    # Validate ref_array if provided
    if ref_array is not None:
        if input_array.shape != ref_array.shape:
            raise ValueError(
                f"input_array shape: {input_array.shape} does not match "
                f"ref_array shape: {ref_array.shape}"
            )
    
    # Get dimensions
    n = len(from_resi_idx_list)  # Number of residues to query
    m = len(query_idx_list)      # Number of query regions
    
    # Validate from_resi_idx_list indices
    max_idx = input_array.shape[0]
    for idx in from_resi_idx_list:
        if idx < 1 or idx > max_idx:
            raise ValueError(
                f"Residue index {idx} is out of valid range [1, {max_idx}]"
            )
    
    # Initialize output arrays
    ref_prob = np.zeros((n, m))
    input_prob = np.zeros((n, m))
    diff_prob = np.zeros((n, m))
    
    mean_top_ref_prob = np.zeros((n, m))
    mean_top_input_prob = np.zeros((n, m))
    mean_top_diff_prob = np.zeros((n, m))
    
    top_n_ref_prob = np.zeros((n, top_n * m))
    top_n_input_prob = np.zeros((n, top_n * m))
    top_n_diff_prob = np.zeros((n, top_n * m))
    
    # Iterate through each residue in from_resi_idx_list
    for i, resi_idx in enumerate(from_resi_idx_list):
        # Convert 1-based to 0-based index
        row_idx = resi_idx - 1
        
        # Iterate through each query region
        for j, (q_start_idx, q_end_idx) in enumerate(query_idx_list):
            # Extract submatrix for current residue vs current query region
            input_sub_mat = input_array[row_idx:row_idx+1, q_start_idx:q_end_idx]
            
            # Calculate maximum contact probabilities
            input_prob_val = np.round(np.max(input_sub_mat), 4)
            input_prob[i, j] = input_prob_val
            
            # Calculate top-n averaged contact probabilities
            mean_top_input_val = np.round(top_n_average(input_sub_mat, top_n), 4)
            mean_top_input_prob[i, j] = mean_top_input_val
            
            # Calculate top-n individual contact probabilities
            top_n_input_val = np.round(get_top_n(input_sub_mat, top_n), 4)
            # Store in the appropriate columns (top_n values per query region)
            top_n_input_prob[i, j*top_n:(j+1)*top_n] = top_n_input_val
            
            # Process ref_array if provided
            if ref_array is not None:
                ref_sub_mat = ref_array[row_idx:row_idx+1, q_start_idx:q_end_idx]
                
                # Maximum contact probabilities
                ref_prob_val = np.round(np.max(ref_sub_mat), 4)
                ref_prob[i, j] = ref_prob_val
                diff_prob[i, j] = np.round(input_prob_val - ref_prob_val, 4)
                
                # Top-n averaged contact probabilities
                mean_top_ref_val = np.round(top_n_average(ref_sub_mat, top_n), 4)
                mean_top_ref_prob[i, j] = mean_top_ref_val
                mean_top_diff_prob[i, j] = np.round(mean_top_input_val - mean_top_ref_val, 4)
                
                # Top-n individual contact probabilities
                top_n_ref_val = np.round(get_top_n(ref_sub_mat, top_n), 4)
                top_n_ref_prob[i, j*top_n:(j+1)*top_n] = top_n_ref_val
                top_n_diff_prob[i, j*top_n:(j+1)*top_n] = np.round(
                    top_n_input_val - top_n_ref_val, 4
                )
    
    # Prepare output dictionary
    out_prob_dict = {
        "ref_prob": ref_prob,
        "input_prob": input_prob,
        "diff_prob": diff_prob,
        "mean_top_ref_prob": mean_top_ref_prob,
        "mean_top_input_prob": mean_top_input_prob,
        "mean_top_diff_prob": mean_top_diff_prob,
        "top_n_ref_prob": top_n_ref_prob,
        "top_n_input_prob": top_n_input_prob,
        "top_n_diff_prob": top_n_diff_prob
    }
    
    return out_prob_dict


####################################################################################
# Original function (kept for backward compatibility)
####################################################################################
def query_resi_to_nuc_contact_top_n(
    on_array: np.ndarray,
    off_array: np.ndarray,
    query_idx_list: list = [(1368, 1368+23)],
    top_n: int = 3,
    cas_length: int = 1368
) -> dict:
    """
    Compare contact probability between on-target and off-target for Cas protein residues.
    
    [Original docstring preserved...]
    This is the original function kept for backward compatibility.
    For new code, consider using query_cp_with_top_n() instead.
    """
    
    # Initialize output dictionary
    out_prob_dict = {
        "on_prob": [],
        "off_prob": [],
        "diff_prob": [],
        "top_on_prob": [],
        "top_off_prob": [],
        "top_diff_prob": [],
        "top_n_on_prob": [],
        "top_n_off_prob": [],
        "top_n_diff_prob": []
    }
    
    # Assign matrices
    on_mat = on_array
    off_mat = off_array
        
    # Validate matrix shapes
    if on_mat.shape != off_mat.shape:
        raise ValueError(
            f"on_mat shape: {on_mat.shape} does not match off_mat shape: {off_mat.shape}"
        )
    
    # Iterate through each Cas protein residue
    for resi_idx in range(1, cas_length + 1):
        # Create masks for row (residue) and columns (query regions)
        row_mask = np.zeros(on_mat.shape[0], dtype=bool)
        col_mask = np.zeros(on_mat.shape[1], dtype=bool)
        
        # Set row mask for current residue
        row_mask[resi_idx - 1] = True
        
        # Set column mask for all query regions
        for q_start_idx, q_end_idx in query_idx_list:
            col_mask[q_start_idx:q_end_idx] = True
        
        # Extract submatrices for current residue vs query regions
        on_sub_mat = on_mat[row_mask][:, col_mask]
        off_sub_mat = off_mat[row_mask][:, col_mask]
        
        # Calculate maximum contact probabilities (top 1)
        on_prob_val = np.round(np.max(on_sub_mat), 4)
        off_prob_val = np.round(np.max(off_sub_mat), 4)
        diff_prob_val = np.round(off_prob_val - on_prob_val, 4)
        out_prob_dict["on_prob"].append(on_prob_val)
        out_prob_dict["off_prob"].append(off_prob_val)
        out_prob_dict["diff_prob"].append(diff_prob_val)
        
        # Calculate top-n averaged contact probabilities
        top_on_prob_val = np.round(top_n_average(on_sub_mat, top_n), 4)
        top_off_prob_val = np.round(top_n_average(off_sub_mat, top_n), 4)
        top_diff_prob_val = np.round(top_off_prob_val - top_on_prob_val, 4)
        out_prob_dict["top_on_prob"].append(top_on_prob_val)
        out_prob_dict["top_off_prob"].append(top_off_prob_val)
        out_prob_dict["top_diff_prob"].append(top_diff_prob_val)
        
        # Calculate top-n individual contact probabilities (as arrays)
        top_n_on_prob_val = np.round(get_top_n(on_sub_mat, top_n), 4)
        top_n_off_prob_val = np.round(get_top_n(off_sub_mat, top_n), 4)
        top_n_diff_prob_val = np.round(top_n_off_prob_val - top_n_on_prob_val, 4) 
        out_prob_dict["top_n_on_prob"].append(top_n_on_prob_val)
        out_prob_dict["top_n_off_prob"].append(top_n_off_prob_val)
        out_prob_dict["top_n_diff_prob"].append(top_n_diff_prob_val)

    # Convert all lists to numpy arrays
    out_prob_dict["on_prob"] = np.array(out_prob_dict["on_prob"])
    out_prob_dict["off_prob"] = np.array(out_prob_dict["off_prob"])
    out_prob_dict["diff_prob"] = np.array(out_prob_dict["diff_prob"])
    out_prob_dict["top_on_prob"] = np.array(out_prob_dict["top_on_prob"])
    out_prob_dict["top_off_prob"] = np.array(out_prob_dict["top_off_prob"])
    out_prob_dict["top_diff_prob"] = np.array(out_prob_dict["top_diff_prob"])
    out_prob_dict["top_n_on_prob"] = np.array(out_prob_dict["top_n_on_prob"])
    out_prob_dict["top_n_off_prob"] = np.array(out_prob_dict["top_n_off_prob"])
    out_prob_dict["top_n_diff_prob"] = np.array(out_prob_dict["top_n_diff_prob"])
    
    return out_prob_dict


####################################################################################
# Demo and testing code
####################################################################################
if __name__ == "__main__":
    print("Testing query_cp_with_top_n function...")
    
    # Create test matrices
    np.random.seed(42)
    matrix_size = 1500
    input_mat = np.random.rand(matrix_size, matrix_size)
    ref_mat = np.random.rand(matrix_size, matrix_size)
    
    # Test parameters
    from_resi = [1, 10, 100, 500, 1000]  # 5 residues (n=5)
    query_regions = [(1368, 1391), (1400, 1420)]  # 2 query regions (m=2)
    top_n = 3
    
    print(f"\nTest configuration:")
    print(f"  Matrix size: {matrix_size} × {matrix_size}")
    print(f"  Number of residues (n): {len(from_resi)}")
    print(f"  Number of query regions (m): {len(query_regions)}")
    print(f"  Top-n: {top_n}")
    
    # Test with ref_array
    print("\n--- Test 1: With ref_array ---")
    result_with_ref = query_cp_with_top_n(
        input_array=input_mat,
        from_resi_idx_list=from_resi,
        query_idx_list=query_regions,
        top_n=top_n,
        ref_array=ref_mat
    )
    
    print(f"\nOutput shapes:")
    print(f"  input_prob: {result_with_ref['input_prob'].shape} (expected: {len(from_resi)}, {len(query_regions)})")
    print(f"  ref_prob: {result_with_ref['ref_prob'].shape}")
    print(f"  diff_prob: {result_with_ref['diff_prob'].shape}")
    print(f"  mean_top_input_prob: {result_with_ref['mean_top_input_prob'].shape}")
    print(f"  top_n_input_prob: {result_with_ref['top_n_input_prob'].shape} (expected: {len(from_resi)}, {top_n * len(query_regions)})")
    
    print(f"\nSample values (first residue, first query region):")
    print(f"  Max input_prob: {result_with_ref['input_prob'][0, 0]}")
    print(f"  Max ref_prob: {result_with_ref['ref_prob'][0, 0]}")
    print(f"  Diff: {result_with_ref['diff_prob'][0, 0]}")
    print(f"  Mean top-n input: {result_with_ref['mean_top_input_prob'][0, 0]}")
    print(f"  Top-n values (input): {result_with_ref['top_n_input_prob'][0, :top_n]}")
    
    # Test without ref_array
    print("\n--- Test 2: Without ref_array (all diff values should be 0) ---")
    result_without_ref = query_cp_with_top_n(
        input_array=input_mat,
        from_resi_idx_list=from_resi,
        query_idx_list=query_regions,
        top_n=top_n,
        ref_array=None
    )
    
    print(f"\nOutput shapes:")
    print(f"  input_prob: {result_without_ref['input_prob'].shape}")
    print(f"  ref_prob: {result_without_ref['ref_prob'].shape} (should be all zeros)")
    print(f"  diff_prob: {result_without_ref['diff_prob'].shape} (should be all zeros)")
    
    print(f"\nVerifying all diff values are zero:")
    print(f"  diff_prob all zeros: {np.all(result_without_ref['diff_prob'] == 0)}")
    print(f"  mean_top_diff_prob all zeros: {np.all(result_without_ref['mean_top_diff_prob'] == 0)}")
    print(f"  top_n_diff_prob all zeros: {np.all(result_without_ref['top_n_diff_prob'] == 0)}")
    
    print("\n✓ All tests completed!")
