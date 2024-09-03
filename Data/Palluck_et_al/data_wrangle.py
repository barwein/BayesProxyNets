import numpy as np
import pandas as pd
import pyreadr
import re
import jax.numpy as jnp
from itertools import combinations
import torch
import networkx as nx


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


def network_by_school(df: pd.DataFrame, school_id: float | int, cols: list[str], plot_network = False) -> np.ndarray:
    """
    Create network adjacency matrix for a given school.
    :param df: data frame
    :param school_id: SCHID value
    :param cols: list of columns that defines the network
    :return: adj matrix
    """
    # Save subset of df
    school_mask = df['SCHID'] == school_id
    school_df = df[school_mask].copy()  # Create a copy to avoid warnings
    # school_df['unique_id'] = (school_df['SCHID'] * 1000 + school_df['ID']).astype(int)
    # school_df['unique_id'] = school_df['unique_id'].astype(int)
    # Replace 999 with NaN as 999 is the code for missing values
    for col in cols:
        row_mask = school_df[col] == 999
        school_df.loc[row_mask, col] = np.nan
    # Add SCHID * 1000 to survey data (to obtain edge list with `unique_id` values)
    for col in cols:
        school_df[col] += school_df['SCHID'] * 1000
    # Save edgelists
    school_edgelist = []
    for col in cols:
        school_edgelist.extend(zip(school_df['unique_id'], school_df[col]))

    valid_ids = set(school_df['unique_id'])
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
                         node_color=school_df['TREAT_NUMERIC'],
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


def create_net_covar_df(df: pd.DataFrame, schid: float) -> torch.tensor:
    """
    Create a data frame with covariates for network analysis.
    :param df: data frame
    :param cov_groups: which covariates to include
    :return: data frame of covariates ready for network analysis
    """
    # Save subset of df
    school_mask = df['SCHID'] == schid
    school_df = df[school_mask].copy()  # Create a copy to avoid warnings
    idx_pairs = list(combinations(range(school_df.shape[0]), 2))
    cov_eq = [cov_equal(school_df[cov], idx_pairs) for cov in COV_FOR_NETWORK]
    df_network = pd.DataFrame(dict(zip(['+'.join(cov) for cov in COV_FOR_NETWORK], cov_eq)))

    expected_rows = school_df.shape[0] * (school_df.shape[0] - 1) // 2
    assert df_network.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df_network.shape[0]}"

    return torch.tensor(np.array(df_network), dtype=torch.float32)

def adj_to_triu(mat: np.ndarray) -> torch.tensor:
    """
    Convert adjacency matrix to upper triangular matrix.
    :param mat: adj matrix
    :return: triu ('upper triangle') values
    """
    return torch.tensor(mat[np.triu_indices(mat.shape[0], k=1)], dtype=torch.float32)

