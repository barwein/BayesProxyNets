import numpy as np
import pandas as pd
import pyreadr
import re
import jax.numpy as jnp
from itertools import combinations
import torch
import networkx as nx
import utils_for_inference as util


### Global variables ###
REL_VARIABLES = ['SCHID', 'SCHTREAT_NUMERIC', 'TREAT_NUMERIC', 'unique_id', 'ELIGIBLE', 'WRISTOW2_NUMERIC']
COV_LIST = ["GENC", "GRC", "ETHW", "ETHB", "ETHH", "ETHA", "ETHC", "ETHSA", "GAME"]
ST_COLS = ['ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8', 'ST9', 'ST10']
ST_W2_COLS = ['ST1W2', 'ST2W2', 'ST3W2', 'ST4W2', 'ST5W2', 'ST6W2', 'ST7W2', 'ST8W2', 'ST9W2', 'ST10W2']
BF_COLS = ['BF1', 'BF2']
COV_FOR_NETWORK = [["GENC"],
                   ["ETHW", "ETHB", "ETHH", "ETHA", "ETHC", "ETHSA"],
                   # ["ACTSS", "ACTT", "ACTM", "ACTR"],
                   ["GAME"],
                   ["GRC_6", "GRC_7", "GRC_8"]]

def extract_numeric(x):
    match = re.search(r'\(?([0-9,.]+)\)?', str(x))
    return float(match.group(1)) if match else np.nan


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data from the Palluck et al. (2021) study.
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    # save relevant subset of data
    data_cleaned = data.copy()
    data_cleaned['ID'] = pd.to_numeric(data_cleaned['ID'], errors='coerce').fillna(0.0)
    data_cleaned['SCHRB'] = pd.to_numeric(data_cleaned['SCHRB'], errors='coerce')
    # all_schools = all_schools[~all_schools['SCHRB'].isna() & (all_schools['ID'] != 999.0)]
    data_cleaned = data_cleaned[~data_cleaned['SCHRB'].isna() &
                              (data_cleaned['ID'] != 999.0) &
                              (data_cleaned['ID'] != 0.0) &
                              (data_cleaned['UID'] != 100284.0)]

    # Create unique id
    data_cleaned['unique_id'] = data_cleaned['SCHID'] * 1000 + data_cleaned['ID']
    data_cleaned['unique_id'] = data_cleaned['unique_id'].astype(int)

    # Convert treatment variables to numeric
    data_cleaned['TREAT_NUMERIC'] = data_cleaned['TREAT'].apply(extract_numeric).fillna(0.0).astype(int)
    data_cleaned['SCHTREAT_NUMERIC'] = data_cleaned['SCHTREAT'].apply(extract_numeric).fillna(0.0).astype(int)
    # # Create indicator of eligible units
    data_cleaned['ELIGIBLE'] = (data_cleaned['TREAT_NUMERIC'] != 0.0).fillna(0.0).astype(int)
    # Numeric version of OUTOFBLOCK
    data_cleaned['OUTOFBLOCK_NUMERIC'] = data_cleaned['OUTOFBLOCK'].apply(extract_numeric).fillna(0.0).astype(int)
    # Numeric version of outcome (wearing orange band)
    data_cleaned['WRISTOW2_NUMERIC'] = data_cleaned['WRISTOW2'].apply(extract_numeric).fillna(0.0).astype(int)

    # Get subset of relevant columns for analysis (outcome, treatents, covariates)
    data_subset = data_cleaned[REL_VARIABLES + ST_COLS + ST_W2_COLS + BF_COLS]
    for cov in COV_LIST:
        if type(data_cleaned[cov].iloc[1]) == str:
            val = data_cleaned[cov].apply(extract_numeric).fillna(0.0).astype(int)
            # df_subset.loc[:,cov] = val
            data_subset = data_subset.assign(**{cov: val})
        else:
            # df_subset.loc[:,cov] = all_schools[cov]
            data_subset = data_subset.assign(**{cov: data_cleaned[cov]})

    # Get dummies of 'GRC' (grade) variable
    GRC_dummy = pd.get_dummies(data_subset['GRC'], drop_first=True).astype(int)
    GRC_dummy.columns = ['GRC_' + str(col) for col in [6, 7, 8]]
    data_subset = pd.concat([data_subset.drop(columns=['GRC']), GRC_dummy], axis=1)

    return data_subset


def network_by_school(df: pd.DataFrame, cols: list[str], plot_network = False) -> np.ndarray:
    """
    Create network adjacency matrix for a given school.
    :param df: data frame
    :param school_id: SCHID value
    :param cols: list of columns that defines the network
    :return: adj matrix
    """
    # Replace 999 with NaN as 999 is the code for missing values
    for col in cols:
        row_mask = df[col] == 999
        df.loc[row_mask, col] = np.nan
    # Add SCHID * 1000 to survey data (to obtain edge list with `unique_id` values)
    for col in cols:
        # df[col] += df['SCHID'] * 1000
        df.loc[:,col] += df['SCHID'] * 1000
    # Save edgelists
    school_edgelist = []
    for col in cols:
        school_edgelist.extend(zip(df['unique_id'], df[col]))

    valid_ids = set(df['unique_id'])
    school_edgelist = [
        (int(a), int(b))
        for a, b in school_edgelist
        if not np.isnan(a) and not np.isnan(b)
           and int(a) in valid_ids and int(b) in valid_ids
           and int(a) != int(b)
    ]
    # Add selfloops
    for id in valid_ids:
        school_edgelist.append((id, id))
    # Convert to nx graph
    school_network = nx.Graph(school_edgelist)
    # remove self loops
    school_network.remove_edges_from(nx.selfloop_edges(school_network))

    if plot_network:
        nx.draw_circular(school_network,
                         node_color=df['TREAT_NUMERIC'],
                         node_size=[school_network.degree(node) + 1 for node in school_network.nodes()],
                         width=0.15)
    # return adj. matrix
    return nx.to_numpy_array(school_network)


def cov_equal(X: pd.DataFrame, idx_pairs: list) -> list[int]:
    """
    Create a list of binary values indicating whether the covariates are equal for each pair of units.
    :param X: data frame
    :param idx_pairs: list of indices of pairs
    :return: list of binary values
    """
    return [int(np.all(X.iloc[i] == X.iloc[j])) for i, j in idx_pairs]


def create_net_covar_df(df: pd.DataFrame) -> torch.tensor:
    """
    Create a data frame with covariates for network analysis.
    :param df: data frame
    :return: data frame of covariates ready for network analysis
    """
    # Save subset of df
    idx_pairs = list(combinations(range(df.shape[0]), 2))
    cov_eq = [cov_equal(df[cov], idx_pairs) for cov in COV_FOR_NETWORK]
    df_network = pd.DataFrame(dict(zip(['+'.join(cov) for cov in COV_FOR_NETWORK], cov_eq)))

    expected_rows = df.shape[0] * (df.shape[0] - 1) // 2
    assert df_network.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df_network.shape[0]}"

    return torch.tensor(np.array(df_network), dtype=torch.float32)

def adj_to_triu(mat: np.ndarray) -> torch.tensor:
    """
    Convert adjacency matrix to upper triangular matrix.
    :param mat: adj matrix
    :return: triu ('upper triangle') values
    """
    return torch.tensor(mat[np.triu_indices(mat.shape[0], k=1)], dtype=torch.float32)


def group_indicators_to_indices(df):
    """
    Convert a DataFrame of binary group indicators to a vector of indices.
    Parameters:
    df (pandas.DataFrame): N x K DataFrame of binary indicators.
                           Each row should sum to 1.
    Returns:
    jax.numpy.ndarray: N x 1 array of indices.
    """
    # Check if each row sums to 1
    if not (df.sum(axis=1) == 1).all():
        raise ValueError("Each row in the DataFrame must sum to 1")
    indices = jnp.argmax(df.values, axis=1)
    return indices

def data_for_outcome_regression(df, adj_mat):
    """Get the school data for the outcome regression"""
    school_df_elig = df[df['ELIGIBLE'] == 1]
    fixed_df = jnp.array(school_df_elig[["GENC", "ETHW", "ETHH", "GAME"]].values)
    all_trts = jnp.array(df['TREAT_NUMERIC'].values == 2, dtype=int)
    elig_trts = jnp.array(school_df_elig['TREAT_NUMERIC'].values == 2, dtype=int)
    Y = jnp.array(school_df_elig['WRISTOW2_NUMERIC'].values)
    sch_trt = jnp.array(school_df_elig['SCHTREAT_NUMERIC'].values)
    exposures_all = util.zeigen_value(all_trts, adj_mat)
    exposures_elig = exposures_all[(df['ELIGIBLE'] == 1).values]
    grades = group_indicators_to_indices(school_df_elig[['GRC_6', 'GRC_7', 'GRC_8']])
    school = jnp.array(school_df_elig['SCHID'].values, dtype = int)
    # school = jnp.array(school_df_elig['SCHID'].values)

    return {'X' : fixed_df, 'school' : school,  'grade' : grades,
            'trts' : elig_trts, 'sch_trts' : sch_trt ,
             'exposures' : exposures_elig, 'Y' : Y}


def concatenate_dict_arrays(dict_list):
    """
    Concatenate arrays from a list of dictionaries.
    Each dictionary contains arrays with shapes:
    [(n_x, 4), (n_x,), (n_x,), (n_x,), (n_x,), (n_x,), (n_x,)]

    :param dict_list: List of dictionaries containing JAX arrays
    :return: A single dictionary with concatenated arrays
    """
    # Initialize the result dictionary
    result = {}
    # Get the keys from the first dictionary (assuming all dictionaries have the same keys)
    keys = dict_list[0].keys()
    for key in keys:
        # Collect all arrays for this key across all dictionaries
        arrays_to_concat = [d[key] for d in dict_list]
        result[key] = jnp.concatenate(arrays_to_concat)

    return result


def transform_schid(schid_array):
    """
    Transform school IDs to be in range (0, n_unique_schid-1).

    :param schid_array: JAX array of shape (n,) containing school IDs
    :return: JAX array of same shape with transformed school IDs
    """
    unique_ids, inverse_indices = jnp.unique(schid_array, return_inverse=True)
    return inverse_indices
