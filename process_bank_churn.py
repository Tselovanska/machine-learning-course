"""
process_bank_churn.py

Modular preprocessing pipeline for the bank churn dataset.
Exposes preprocess_data() for train/val prep and preprocess_new_data()
for transforming unseen data (e.g. test.csv) with already-fitted
scaler/encoder before feeding a model (e.g. a decision tree).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, List, Tuple, Optional


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names for consistency.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: DataFrame with lowercase column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def drop_irrelevant_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop columns that carry no predictive signal (e.g. id, surname, customerid).

    Args:
        df (pd.DataFrame): Dataframe to drop columns from.
        columns (List[str]): Column names to drop. Missing names are ignored.

    Returns:
        pd.DataFrame: DataFrame without the specified columns.
    """
    return df.drop(columns=columns, errors='ignore')


def split_data_stratified(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Split a dataframe into train/val sets, preserving the class ratio of target_col.

    Args:
        df (pd.DataFrame): Dataframe to split.
        target_col (str): Column to stratify on (imbalanced target).
        test_size (float): Fraction of data allocated to validation.
        random_state (int): Seed for reproducibility.

    Returns:
        Dict[str, pd.DataFrame]: {'train': train_df, 'val': val_df}.
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    return {'train': train_df, 'val': val_df}


def create_inputs_targets(
    df_dict: Dict[str, pd.DataFrame],
    input_cols: List[str],
    target_col: str
) -> Dict[str, Any]:
    """
    Split each dataframe in df_dict into inputs (features) and targets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): {'train': train_df, 'val': val_df}.
        input_cols (List[str]): Feature column names.
        target_col (str): Target column name.

    Returns:
        Dict[str, Any]: {'train_inputs', 'train_targets', 'val_inputs', 'val_targets'}.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def get_numeric_and_categorical_cols(inputs: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical (object dtype) columns in a dataframe.

    Args:
        inputs (pd.DataFrame): Feature dataframe.

    Returns:
        Tuple[List[str], List[str]]: (numeric_cols, categorical_cols).
    """
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes('object').columns.tolist()
    return numeric_cols, categorical_cols


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training numeric columns.

    Args:
        train_inputs (pd.DataFrame): Training feature dataframe.
        numeric_cols (List[str]): Numeric column names to fit on.

    Returns:
        MinMaxScaler: Fitted scaler.
    """
    return MinMaxScaler().fit(train_inputs[numeric_cols])


def apply_scaler(inputs: pd.DataFrame, numeric_cols: List[str], scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Scale numeric columns of a dataframe into the 0..1 range using a fitted scaler.

    Args:
        inputs (pd.DataFrame): Feature dataframe to transform.
        numeric_cols (List[str]): Numeric column names to scale.
        scaler (MinMaxScaler): Previously fitted scaler.

    Returns:
        pd.DataFrame: DataFrame with numeric_cols scaled (modified copy).
    """
    inputs = inputs.copy()
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    return inputs


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on the training categorical columns.

    Args:
        train_inputs (pd.DataFrame): Training feature dataframe.
        categorical_cols (List[str]): Categorical column names to fit on.

    Returns:
        OneHotEncoder: Fitted encoder.
    """
    return OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])


def apply_encoder(
    inputs: pd.DataFrame,
    categorical_cols: List[str],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using a fitted encoder and drop the originals.

    Args:
        inputs (pd.DataFrame): Feature dataframe to transform.
        categorical_cols (List[str]): Categorical column names to encode.
        encoder (OneHotEncoder): Previously fitted encoder.

    Returns:
        pd.DataFrame: DataFrame with categorical_cols replaced by one-hot columns.
    """
    inputs = inputs.copy()
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded = encoder.transform(inputs[categorical_cols])
    inputs[encoded_cols] = encoded
    inputs = inputs.drop(columns=categorical_cols)
    return inputs


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str = 'exited',
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run the full preprocessing pipeline on raw training data: normalize columns,
    drop non-predictive columns, stratified train/val split, fit encoder/scaler
    on train, and transform both splits.

    Args:
        raw_df (pd.DataFrame): Raw dataframe (e.g. train.csv contents).
        target_col (str): Target column name.
        id_cols (Optional[List[str]]): Non-predictive columns to drop
            (default: ['id', 'surname', 'customerid']).
        test_size (float): Fraction of data allocated to validation.
        random_state (int): Seed for reproducibility.

    Returns:
        Dict[str, Any]: {
            'X_train': pd.DataFrame,
            'train_targets': pd.Series,
            'X_val': pd.DataFrame,
            'val_targets': pd.Series,
            'input_cols': List[str],
            'numeric_cols': List[str],
            'categorical_cols': List[str],
            'scaler': MinMaxScaler,
            'encoder': OneHotEncoder,
        }
    """
    if id_cols is None:
        id_cols = ['id', 'surname', 'customerid']

    df = normalize_columns(raw_df)
    df = drop_irrelevant_columns(df, id_cols)

    split_dfs = split_data_stratified(df, target_col, test_size, random_state)
    input_cols = list(df.columns.drop(target_col))
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols, categorical_cols = get_numeric_and_categorical_cols(data['train_inputs'])

    scaler = fit_scaler(data['train_inputs'], numeric_cols)
    encoder = fit_encoder(data['train_inputs'], categorical_cols)

    train_inputs = apply_scaler(data['train_inputs'], numeric_cols, scaler)
    train_inputs = apply_encoder(train_inputs, categorical_cols, encoder)

    val_inputs = apply_scaler(data['val_inputs'], numeric_cols, scaler)
    val_inputs = apply_encoder(val_inputs, categorical_cols, encoder)

    return {
        'X_train': train_inputs,
        'train_targets': data['train_targets'],
        'X_val': val_inputs,
        'val_targets': data['val_targets'],
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'scaler': scaler,
        'encoder': encoder,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    id_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocess unseen data (e.g. test.csv) using an already-fitted scaler and
    encoder from preprocess_data(), for prediction/evaluation.

    Args:
        new_df (pd.DataFrame): Raw new dataframe (no target column required).
        input_cols (List[str]): Feature column names, as returned by preprocess_data().
        numeric_cols (List[str]): Numeric column names, as returned by preprocess_data().
        categorical_cols (List[str]): Categorical column names, as returned by preprocess_data().
        scaler (MinMaxScaler): Scaler fitted on training data.
        encoder (OneHotEncoder): Encoder fitted on training data.
        id_cols (Optional[List[str]]): Non-predictive columns to drop
            (default: ['id', 'surname', 'customerid']).

    Returns:
        pd.DataFrame: Transformed feature dataframe, ready for model inference.
    """
    if id_cols is None:
        id_cols = ['id', 'surname', 'customerid']

    df = normalize_columns(new_df)
    df = drop_irrelevant_columns(df, id_cols)

    inputs = df[input_cols].copy()
    inputs = apply_scaler(inputs, numeric_cols, scaler)
    inputs = apply_encoder(inputs, categorical_cols, encoder)

    return inputs
