import pandas as pd
import numpy as np


def load_dataset(dataset_name, input_size, output_size, header=None):
    """
    Loads a dataset from a CSV file, assigns column names, 
    and splits it into input and output data.

    Parameters:
    dataset_name (str): Path to the CSV file.
    input_size (int): Number of columns to include in the input data.
    output_size (int): Number of columns to include in the output data.
    header (int or None, optional): Row number to use as the column names. 
                                    Default is None, meaning no header row.

    Returns:
    tuple: A tuple containing:
        - input_data (pd.DataFrame): DataFrame with the first `input_size` columns.
        - output_data (pd.DataFrame): DataFrame with the last `output_size` columns.
    """

    # Load the dataset from the CSV file
    df = pd.read_csv(dataset_name, header=header)

    # Assign default column names (e.g., col1, col2, ..., colN)
    df.columns = [f"col{i}" for i in range(1, df.shape[1] + 1)]

    # Split the DataFrame into input and output parts
    input_data = df.iloc[:, :input_size]  # Select the first `input_size` columns
    output_data = df.iloc[:, -output_size:]  # Select the last `output_size` columns

    # Return the input and output DataFrames
    return input_data, output_data



def create_datasets_matrices(input_data, output_data, num_rows):
    """
    Extracts the first `num_rows` rows from the input and output data 
    and converts them into numpy matrices.

    Parameters:
    input_data (pd.DataFrame): DataFrame containing the input features.
    output_data (pd.DataFrame): DataFrame containing the output targets.
    num_rows (int): Number of rows to select for both input and output.

    Returns:
    tuple: A tuple containing:
        - input_matrix (numpy.ndarray): Matrix of shape (num_rows, num_columns_input).
        - output_matrix (numpy.ndarray): Matrix of shape (num_rows, num_columns_output).
    """

    # Select the first `num_rows` rows from the input and output data
    input_data = input_data.iloc[:num_rows, :] 
    output_data = output_data.iloc[:num_rows, :]  

    # Convert the DataFrames to numpy arrays
    input_matrix = input_data.values  
    output_matrix = output_data.values 

    return input_matrix, output_matrix
