# -*- coding: UTF-8 -*-

import numpy as np


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
# function define [top-n extraction]
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
    
    Analyzes contact probability matrix to extract and compare interactions between
    Cas protein residues and specified nucleic acid regions (queries). For each Cas residue,
    calculates maximum contact probability, top-n average, and top-n individual values.
    
    Args:
        on_array: Contact probability matrix for on-target. 

        off_array: Contact probability matrix for off-target.
                   Must have the same shape as on_array.
                   
        query_idx_list: List of tuples specifying nucleic acid query regions in Python-style
                        indexing (start, end). Default is [(1368, 1391)] representing a 
                        23-nucleotide region starting at index 1368.
                        
        top_n: Number of top contact values to extract for averaging and analysis. Default is 3.
        
        cas_length: Number of Cas protein residues (token length). Default is 1368.
                    Indices 0 to cas_length-1 in the matrices represent Cas residues.
    
    Returns:
        A dictionary containing contact probability analyses with the following keys:
            - "on_prob": Array of maximum contact probabilities for each residue (on-target).
                        Shape: (cas_length,)
            - "off_prob": Array of maximum contact probabilities for each residue (off-target).
                         Shape: (cas_length,)
            - "diff_prob": Array of differences (off - on) in maximum contact probabilities.
                          Shape: (cas_length,)
            - "top_on_prob": Array of top-n averaged contact probabilities (on-target).
                            Shape: (cas_length,)
            - "top_off_prob": Array of top-n averaged contact probabilities (off-target).
                             Shape: (cas_length,)
            - "top_diff_prob": Array of differences (off - on) in top-n averaged probabilities.
                              Shape: (cas_length,)
            - "top_n_on_prob": 2D array containing top n contact values (on-target).
                              Shape: (cas_length, top_n)
            - "top_n_off_prob": 2D array containing top n contact values (off-target).
                               Shape: (cas_length, top_n)
            - "top_n_diff_prob": 2D array of differences (off - on) in top n values.
                                Shape: (cas_length, top_n)
        
        Each list contains cas_length elements, one for each Cas protein residue.
    
    Raises:
        ValueError: If on_array and off_array have different shapes.
    
    Note:
        All probability values are rounded to 4 decimal places. The function calculates
        three types of metrics: (1) maximum value, (2) average of top-n values, and 
        (3) individual top-n values as arrays.
    
    Example:
        >>> on_mat = np.random.rand(1391, 1391)
        >>> off_mat = np.random.rand(1391, 1391)
        >>> result = query_resi_to_nuc_contact_off_vs_on(on_mat, off_mat, top_n=3)
        >>> len(result["on_prob"])
        1368
        >>> len(result["top_n_on_prob"][0])
        3
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



    