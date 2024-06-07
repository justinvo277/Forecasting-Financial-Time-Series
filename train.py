import torch
import random
import argparse
import numpy as np
import pandas as pd

from dataloader import TransformerDataset
from utils import get_indices_entire_sequence, generate_square_subsequent_mask
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe
from transformer_model import TimeSeriesTransformer
from torch.utils.data import DataLoader



parser = argparse.ArgumentParser(description="Config")
parser.add_argument("--data_path", type=str, help="Path of dataset", default="D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data\data_csv\FPT_stock.csv")
parser.add_argument("--datafile_type", type=str, help="csv, xlsx, ....", default="csv")
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
    print(len(dataloader))

    #Define batch;
    batch_size = 32
    train_data = DataLoader(dataset=dataloader, batch_size=batch_size)
    print(f"number of batch: {len(train_data)}")

    i, batch = next(enumerate(train_data))
    src, trg, trg_y = batch
    src=src.float()
    trg=trg.float()

    #Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
    batch_first = False
    if batch_first == False:

        shape_before = src.shape
        src = src.permute(1, 0, 2)
        print("src shape changed from {} to {}".format(shape_before, src.shape))

        shape_before = trg.shape
        trg = trg.permute(1, 0, 2)
        print("src shape changed from {} to {}".format(shape_before, src.shape))

    #Model
    model = TimeSeriesTransformer(
        input_size=8,
        dec_seq_len=30,
        batch_first=batch_first,
        num_predicted_features=8
        )
    model=model.float()

    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = generate_square_subsequent_mask(
        dim1=5,
        dim2=30
        )

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = generate_square_subsequent_mask( 
        dim1=5,
        dim2=5
    )

    output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )