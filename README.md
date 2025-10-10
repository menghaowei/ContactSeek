# ContactSeek

ContactSeek is a computational framework for identifying and ranking protein-nucleotide contact regions using AlphaFold3 contact probability matrices.

## 1. Environment Installation and Configuration

### Method 1: Using micromamba

```bash
micromamba create -n ContactSeek dill numpy pandas matplotlib scipy scikit-learn tqdm
```

### Method 2: Using conda with environment file

This method uses the `environment_ContactSeek.yaml` file to create the Conda environment. This will not work for MacOS.

```bash
conda env create -f environment_ContactSeek.yaml -n ContactSeek`
```

## 2. Loading AlphaFold3 Contact Probability Matrix

Each AlphaFold3 prediction generates two key files:

- A `.cif` file, which contains the predicted 3D structure of the molecular complex.

- A `_confidences.json` file, which stores quality control metrics and, importantly, the inter-token **contact probability (CP)** matrix.

The following example demonstrates how to load and manipulate the CP matrix from the `.json` file.

### 2.1 Loading the CP Matrix

First, we load the necessary libraries and the `.json` file into a NumPy array.

```Python
import numpy as np
import json

json_filename = "./data/spcas9_abe8e_on_af3__confidences.json"

with open(json_filename, "r") as af3_j_file:
    af3_json = json.load(af3_j_file)

# Load contact probabilities as a numpy array
cp_mat = np.array(af3_json["contact_probs"])

print(type(cp_mat))
# Expected output: <class 'numpy.ndarray'>
```

### 2.2 Slicing the CP Matrix to Extract Sub-matrices

Once loaded, the matrix can be sliced to isolate contacts between specific chains or regions of interest. Here we define the sequence lengths and indices for a SpCas9-sgRNA-DNA complex.

**Prediction Information:**

- **Chain A**: SpCas9 (1368 residues)

- **Chain B**: sgRNA (100 nucleotides)

- **Chain C**: Target strand DNA (tsDNA) (43 nucleotides)

- **Chain D**: Non-target strand DNA (ntsDNA) (43 nucleotides)

- **Total Tokens**: 1368 + 100 + 43 + 43 = 1554

```Python
# Extract the Cas9-Cas9 contact submatrix
cas_sub_cp_mat = cp_mat[0:1368, 0:1368]
print(cas_sub_cp_mat.shape)
# Expected output: (1368, 1368)

# Extract the sgRNA-tsDNA contact submatrix in the spacer region
sgRNA_start_idx = 1368
sgRNA_end_idx = 1388
tsDNA_start_idx = 1481
tsDNA_end_idx = 1501
sg_ts_sub_cp_mat = cp_mat[sgRNA_start_idx:sgRNA_end_idx, tsDNA_start_idx:tsDNA_end_idx]
print(sg_ts_sub_cp_mat.shape)
# Expected output: (20, 20)

# Extract Cas9 to all nucleic acid contacts
cas_start_idx = 0
cas_end_idx = 1368
nuc_start_idx = 1368
nuc_end_idx = 1554
cas_nuc_sub_cp_mat = cp_mat[cas_start_idx:cas_end_idx, nuc_start_idx:nuc_end_idx]
print(cas_nuc_sub_cp_mat.shape)
# Expected output: (1368, 186)
```

### 2.3 Calculation of CP Difference (ΔCP)

To compare two different states (e.g., on-target vs. off-target), you can load both CP matrices and calculate the difference.

```Python
# Load on-target and off-target data
on_json_filename = "./data/spcas9_abe8e_on_af3__confidences.json"
off_json_filename = "./data/spcas9_abe8e_off_af3__confidences.json"

with open(on_json_filename, "r") as f:
    on_cp_mat = np.array(json.load(f)["contact_probs"])

with open(off_json_filename, "r") as f:
    off_cp_mat = np.array(json.load(f)["contact_probs"])

# Calculate the difference matrix
cp_diff_mat = off_cp_mat - on_cp_mat

# Example: Check the difference in sgRNA-tsDNA pairing in the PAM distal region
sgRNA_start_idx = 1368
sgRNA_end_idx = 1378
tsDNA_start_idx = 1492
tsDNA_end_idx = 1501
sg_ts_diff_sub_cp_mat = cp_diff_mat[sgRNA_start_idx:sgRNA_end_idx, tsDNA_start_idx:tsDNA_end_idx]

print(sg_ts_diff_sub_cp_mat)
```

Output:

```
array([[ 0.  ,  0.01,  0.06,  0.04,  0.01,  0.  ,  0.01,  0.02, -0.82],
           [ 0.  ,  0.01,  0.02,  0.01,  0.  ,  0.01,  0.03, -0.85, -0.32],
           [ 0.  ,  0.  ,  0.02,  0.01,  0.  ,  0.01, -0.85, -0.21, -0.25],
           [ 0.  ,  0.  ,  0.01,  0.01,  0.  , -0.92, -0.16, -0.23,  0.45],
           [ 0.  ,  0.01,  0.02,  0.02, -0.93, -0.16, -0.17,  0.63,  0.  ],
           [ 0.  ,  0.01,  0.04, -0.9 , -0.29, -0.16,  0.68,  0.01,  0.  ],
           [ 0.01,  0.01, -0.88, -0.88, -0.23,  0.62,  0.  ,  0.  ,  0.  ],
           [ 0.17, -0.69, -0.83, -0.78,  0.28,  0.  ,  0.  ,  0.  ,  0.  ],
           [-0.02, -0.39, -0.78,  0.02,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [-0.02, -0.58,  0.03,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
```

## 3. Top-N Contact Extraction Strategy

ContactSeek includes a "Top-N" strategy to efficiently extract the most significant contact probability for each protein residue with a nucleic acid target. This is useful for feature engineering in downstream analysis.

The function `query_resi_to_nuc_contact_top_n` takes on-target and off-target CP matrices and returns a dictionary containing the top N contacts.

```Python
from ContactSeek.ContactExtraction import query_resi_to_nuc_contact_top_n

# (on_cp_mat and off_cp_mat are loaded as shown in the previous section)

cp_info_dict = query_resi_to_nuc_contact_top_n(
    on_array=on_cp_mat,
    off_array=off_cp_mat,
    query_idx_list=[(1368, 1391)], # Example: query region for nucleic acids
    top_n=3,
    cas_length=1368
)

# The output dictionary contains several keys
# Shape of top N on-target probabilities for each of the 1368 residues
print("Top N on-target shape:", cp_info_dict["top_n_on_prob"].shape)
# Expected output: (1368, 3)

# Shape of top N off-target probabilities
print("Top N off-target shape:", cp_info_dict["top_n_off_prob"].shape)
# Expected output: (1368, 3)

# Shape of the maximum contact probability difference for each residue
print("Max CP difference shape:", cp_info_dict["diff_prob"].shape)
# Expected output: (1368,)
```

## 4. Full Pipeline Demonstration

This section illustrates the complete ContactSeek workflow, from loading data to ranking contact regions and calculating residue-level enhancements.

### 4.1 Dataset Structure

For instance, `data_list_cas9` is a list of length 5, where each element is a dictionary containing data generated from a different AF3 random seed.

Each dictionary follows this structure:

```Python
data_dict = {
    "cp_diff_cas_nuc" : [], # Shape: N x 6 (e.g., 1368 x 6 for Cas9)
    "cp_raw_cas_nuc" : [],  # Shape: N x 18 (e.g., 1368 x 18 for Cas9)
    "cp_raw_on_cas_nuc" : [],# Shape: N x 18 (e.g., 1368 x 18 for Cas9)
    "y_g3" : [],
    "key_info" : []
}
```

All values in the dictionary are lists of the same length (full off-target list), where each element corresponds to a single data sample. 

Here is a detailed description of each key:

- `cp_diff_cas_nuc`: This feature matrix represents the change in contact probability between off-target and on-target predictions.
  
  - For each amino acid, the value in this matrix is the absolute maximum of the contact probability difference (off-target vs. on-target) with one of these six regions.
  
  - The resulting dimension is `N x 6`, where N is the number of amino acids in the protein (e.g., 1368 for SpCas9, 1228 for LbCpf1).

- `cp_raw_cas_nuc`: This matrix contains the raw contact probabilities for the **off-target** predictions.
  
  - For each of the six nucleic acid regions defined above, the top 3 contact probability values for each amino acid are recorded.
  
  - This results in a feature matrix of dimension `N x 18` (N amino acids x 6 regions x 3 top values). For SpCas9, the shape is `1368 x 18`.

- `cp_raw_on_cas_nuc`: This has the same structure as `cp_raw_cas_nuc` (`N x 18`) but contains the top 3 raw contact probabilities for the corresponding **on-target** predictions.

- `y_g3`: This is the target variable for the model. It represents the measured off-target editing efficiency, categorized into three classes: 1, 2, and 3.

### 4.2 Identifying Contact Residues

The `find_contact_residues` function filters residues based on minimum contact probability and difference thresholds.

```Python
from ContactSeek.FindContactResidue import find_contact_residues

contact_residue_cas9 = find_contact_residues(
    merge_cp_raw_list,
    merge_cp_diff_list,
    min_cp_threshold=0.15,
    min_diff_threshold=0.1
)
```

### 4.3 Finding Consensus Contact Regions (CCRs)

Next, `find_consensus_contact_regions` groups the identified contact residues into consensus regions based on their contact profile correlations.

```Python
from ContactSeek.CCRegionFinding import find_consensus_contact_regions
from ContactSeek.ContactPlot import plot_contact_region_correlation

corr_matrix_cas9, kept_indices_cas9, contact_regions_cas9 = find_consensus_contact_regions(
    data_list_cas9,
    contact_residue_cas9,
    correlation_threshold=0.6,
    band_width=7,
    max_merge_iterations=10,
    protein_name='Cas9'
)

# Plot the correlation matrix for a specific region
fig = plot_contact_region_correlation(
    corr_matrix_cas9,
    kept_indices_cas9,
    contact_regions_cas9,
    region_start=660,
    region_end=1000,
    protein_name="Cas9",
    figsize=(10, 10)
)
```

### 4.4 Scoring and Ranking CCRs

The identified CCRs are then scored and ranked based on their importance in predictive models.

```Python
from ContactSeek.Model import main_analysis
from ContactSeek.ContactRanking import ranking_consensus_contact_region
from ContactSeek.Sequence import SEQ_AA_DICT

# Score the regions using machine learning models
models_cas9, results_cas9 = main_analysis(
    data_list_cas9,
    contact_residue_cas9,
    model_performance_log=False
)

# Rank the CCRs based on their scores
ccr_df = ranking_consensus_contact_region(
    contact_regions_cas9,
    models_cas9,
    contact_residue_cas9,
    protein_sequence=SEQ_AA_DICT["spcas9"],
    back_res_dict=False,
)

# Display the top 5 ranked CCRs
print(ccr_df.iloc[:5,])
```

**Output:**

```
   CCR_rank  CCR_ID  CCR_size  CCR_range  raw_score  norm_score                                     CCR_all_residues
0         1      74        10  1030-1039   0.035906    0.943673  1030G,1031K,1032A,1033T,1034A,1035K,1036Y,1037...
1         2      73         7  1023-1029   0.033600    0.892673        1023A,1024K,1025S,1026E,1027Q,1028E,1029I
2         3      72         9  1014-1022   0.029949    0.780605  1014K,1015V,1016Y,1017D,1018V,1019R,1020K,1021...
3         4      62        10   911-920    0.029333    0.760121    911L,912D,913K,914A,915G,916F,917I,918K,919R,920Q
4         5      76         7  1050-1058   0.028439    0.730577    1050I,1051T,1052L,1053A,1054N,1055G,1056E,1057...
```

### 4.5 Calculating Residue-level Contact Enhancement

Finally, we quantified a "contact enhancement".

```Python
from ContactSeek.ContactEnhancement import residue_contact_enhancement

contact_enhance_df = residue_contact_enhancement(
    data_list=data_list_cas9,
    keep_residues=contact_residue_cas9,
    models=models_cas9,
    protein_sequence=SEQ_AA_DICT["spcas9"],
    ccr_ranking_df=ccr_df,
    cpu_threads=90,
    protein_name="Cas9"
)

# Display a sample of the results
print(contact_enhance_df.iloc[150:160,])
```

**Output:**

```
     residue_index residue_name  CCR_index CCR_range  relative_importance  contact_enhance
150            919         919R         62   911-920               2.8313            10.46
151            920         920Q         62   911-920               2.5680             5.59
152            921         921L         63       921               1.6699             0.66
153            922         922V         64       922               2.4016             4.54
154            923         923E         65  923-931                2.3522             2.31
155            924         924T         65  923-931                2.8687             8.54
156            925         925R         65  923-931                2.8676             8.26
157            926         926Q         65  923-931                2.9002             8.47
158            927         927I         65  923-931                2.3349             7.77
159            928         928T         65  923-931                1.3655             0.79
```

## 5. Authors and Contact

- **MENG Haowei**: menghaowei AT gmail.com

- **ZHANG Sihan**: zhangsihann AT gmail.com

- **Supervisor Prof. YI Chengqi**: chengqi.yi AT pku.edu.cn

## 6. License

This project is licensed under the **MIT License**.
