import torch
import random
import argparse
import numpy as np
import pandas as pd

from dataloader import TransformerDataset
from utils.utils import get_indices_entire_sequence
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe



parser = argparse.ArgumentParser(description="Config")
parser.add_argument("--data_path", type=str, help="Path of dataset", default="D:\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data\FPT_stock.xlsx")
parser.add_argument("--datafile_type", type=str, help="csv, xlsx, ....", default="xlsx")
args = parser.parse_args()



if __name__ == "__main__":

    #Read and preprocessing dataset;
    dataset_raw = format_Dataframes(data_path=args.data_path, type_file=args.datafile_type)
    dataset = preprocessing_dataframe(dataset_raw)
    dataset = np.array(dataset)
    print(dataset.shape)
    
    #Dataloader;
    indices_data =get_indices_entire_sequence(data=dataset, window_size=35, step_size=1)
    dataloader = TransformerDataset(
        data=dataset,
        indices= indices_data,
        enc_seq_len= 30,
        dec_seq_len= 5,
        target_seq_len= 5
        )
    
    for i in range(len(dataloader)):
        src, trg, trg_y = dataloader[i]
        print(f"Sample {i+1}:")
        print(f"Source (src): \n{src}\n")
        print(f"Target (trg): \n{trg}\n")
        print(f"Target_y (trg_y): \n{trg_y}\n")
        
        # Limit the print to first 5 samples for brevity
        if i >= 3:
            break
