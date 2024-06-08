import torch
import random
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import TransformerDataset
from transformer_model import TimeSeriesTransformer
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe
from utils import get_indices_entire_sequence, generate_square_subsequent_mask, train_loop, validation_loop

parser = argparse.ArgumentParser(description="Config")
parser.add_argument("--data_path", type=str, help="Path of dataset", default="D:\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data\FPT_stock.xlsx")
parser.add_argument("--log_path", type=str, help="save log for training", default=r"D:\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\log")
parser.add_argument("--datafile_type", type=str, help="csv, xlsx, ....", default="xlsx")

parser.add_argument("--batch_size", type=float, help="Batch size of dataset train, test and val", default=4)
parser.add_argument("--learning_rate", type=float, help="Learning rate for training model", default=1e-5)
parser.add_argument("--batch_first", type=bool, help="Type batch size of dataloader", default=True)
parser.add_argument("--max_viz", type=int, help="After n epochs will validation model", default=1)
parser.add_argument("--epochs", type=int, help="Epochs", default=150)
args = parser.parse_args()



if __name__ == "__main__":

    #Setting device;
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
    DEVICE = 'cpu'

    #Read and preprocessing dataset;
    dataset_raw = format_Dataframes(data_path=args.data_path, type_file=args.datafile_type)
    dataset = preprocessing_dataframe(dataset_raw)
    dataset = np.array(dataset)

    #Dataloader;
    indices_data =get_indices_entire_sequence(data=dataset, window_size=35, step_size=1)
    dataloader = TransformerDataset(
        data=dataset,
        indices= indices_data,
        enc_seq_len= 30,
        dec_seq_len= 5,
        target_seq_len= 5)
    train_data = DataLoader(dataset=dataloader, batch_size=args.batch_size)

    #Model
    model = TimeSeriesTransformer(
        input_size=8,
        dec_seq_len=30,
        batch_first=args.batch_first,
        num_predicted_features=8)
    model = model.float()
    model = model.to(DEVICE)

    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = generate_square_subsequent_mask(dim1=5,dim2=30)
    src_mask = src_mask.to(DEVICE)

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = generate_square_subsequent_mask(dim1=5,dim2=5)
    tgt_mask = tgt_mask.to(DEVICE)


    #Optimizer and criterion;
    opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0, amsgrad=False)
    mse_loss = torch.nn.MSELoss() #Mean Squared Error Loss;
    mae_loss = torch.nn.L1Loss() #Mean Absolute Error Loss;
    huber_loss = torch.nn.SmoothL1Loss() #Huber Loss;
    criterion_list = [('mse', mse_loss), ("mae", mae_loss), ("huber", huber_loss)]


    #Training model;
    for epoch in range(args.epochs):

        print(f"[{epoch}/{args.epochs}: TRAINING PHASE]")
        loss = train_loop(model=model, datatrain=train_data, opt=opt, criterion=mse_loss, 
        epoch=epoch, path_log=args.log_path, src_mask=src_mask, tgt_mask=tgt_mask, device=DEVICE)
        print(f"[{epoch}/{args.epochs}: TRAINING PHASE]: MSE: {loss['MSE']}")

        if epoch % args.max_viz == 0:
            print(f"[{epoch}/{args.epochs}: VALIDATION PHASE]")
            loss_dict = validation_loop(model=model, datatrain=train_data, criterion_list=criterion_list, 
            src_mask=src_mask, tgt_mask=tgt_mask, device=DEVICE)
            print(f"[{epoch}/{args.epochs}: VALIDATION PHASE]: MSE: {loss_dict['mse']} | MAE: {loss_dict['mae']} | HUBER: {loss_dict['huber']}")