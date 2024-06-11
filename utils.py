import os
import torch
import numpy as np
import pandas as pd
import torch.utils
from tqdm import tqdm
from pathlib import Path
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple


def train_loop(model: torch.nn, datatrain: torch.utils.data.Dataset, opt: torch.optim, criterion: dict, 
    epoch: int, path_log: str, src_mask: torch.tensor, tgt_mask: torch.tensor, device: torch.cuda) -> dict:

    loop = tqdm(datatrain, leave=True)
    total_loss = 0

    for batch_idx, (src, trg, trg_y) in enumerate(loop):

        src = src.float()
        src = src.to(device)
        trg = trg.float()
        trg = trg.to(device)
        trg_y = trg_y.float()
        trg_y = trg_y.to(device)

        opt.zero_grad()
        output = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)

        loss = criterion(output, trg_y)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        loss_iter_dict = {"MSE": loss.item()}
        save_log(path_log,  loss_iter_dict, epoch, batch_idx)

        loop.set_postfix(MSE=loss.item())
        
    total_loss /= len(datatrain)
    total_loss_dict = {"MSE LOSS": total_loss}
    save_log(path_log, total_loss_dict, epoch, len(datatrain), f"End  {epoch} epoch")

    return total_loss_dict


def validation_loop(model: torch.nn, datatrain: torch.utils.data.Dataset, criterion_list: dict, 
    src_mask: torch.tensor, tgt_mask: torch.tensor, device: torch.cuda) -> dict:

    loop = tqdm(datatrain, leave=True)
    total_loss_dict = {}

    for name, criterion in criterion_list:
        total_loss_dict[name] = 0.0

    for batch_idx, (src, trg, trg_y) in enumerate(loop):

        src = src.float()
        src = src.to(device)
        trg = trg.float()
        trg = trg.to(device)
        trg_y = trg_y.float()
        trg_y = trg_y.to(device)

        output = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)

        for name, criterion in criterion_list:
            loss = criterion(output, trg_y)
            total_loss_dict[name] += loss.item()
        
    for name_loss in total_loss_dict:
        total_loss_dict[name_loss] /= len(datatrain)

    return total_loss_dict

         
def save_log(folder_log: str, loss_dict, epoch: int, iter: int, mess: str = None) -> None:
    log_name = "log" + str(epoch) + ".txt"
    path_log = os.path.join(folder_log, log_name)

    with open(path_log, 'a') as file:
        if mess != None:
            file.write(f"END {epoch} EPOCH")
        file.write(f"ITERATION {iter} OF EPOCH {epoch} \n")
        file.write("\n")
        file.write(str(loss_dict))

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        """
        Produce all the start and end index positions of all sub-sequences.
        The indices will be used to split the data into sub-sequences on which 
        the models will be trained. 

        Returns a tuple with four elements:
        1) The index position of the first element to be included in the input sequence
        2) The index position of the last element to be included in the input sequence
        3) The index position of the first element to be included in the target sequence
        4) The index position of the last element to be included in the target sequence

        
        Args:
            num_obs (int): Number of observations in the entire dataset for which
                            indices must be generated.

            input_len (int): Length of the input sequence (a sub-sequence of 
                             of the entire data sequence)

            step_size (int): Size of each step as the data sequence is traversed.
                             If 1, the first sub-sequence will be indices 0-input_len, 
                             and the next will be 1-input_len.

            forecast_horizon (int): How many index positions is the target away from
                                    the last index position of the input sequence?
                                    If forecast_horizon=1, and the input sequence
                                    is data[0:10], the target will be data[11:taget_len].

            target_len (int): Length of the target / output sequence.
        """

        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing
        
        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len 
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices


def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences. 

        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences. 
        
        Args:
            num_obs (int): Number of observations (time steps) in the entire 
                           dataset for which indices must be generated, e.g. 
                           len(data)

            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50 
                               time steps, window_size = 100+50 = 150

            step_size (int): Size of each step as the data sequence is traversed 
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size], 
                             and the next will be [1:window_size].

        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        
        subseq_last_idx = window_size
        
        indices = []
        
        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            
            subseq_last_idx += step_size

        return indices


def read_data(data_dir: Union[str, Path] = "data",  
    timestamp_col_name: str="timestamp") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))
    
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path, 
        parse_dates=[timestamp_col_name], 
        index_col=[timestamp_col_name], 
        infer_datetime_format=True,
        low_memory=False
    )

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data


def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df


# if __name__ == "__main__":
#     read_data = read_data(data_dir="D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data\data_csv",
#                           timestamp_col_name="Ngày")
#     print(read_data)