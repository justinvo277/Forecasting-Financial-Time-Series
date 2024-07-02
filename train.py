import torch
import wandb
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import TransformerDataset
from transformer_model import TimeSeriesTransformer
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe, split_data, remove_outliers, winsorize_dataframe
from utils import get_indices_entire_sequence, generate_square_subsequent_mask, train_loop, validation_loop, EarlyStopping
from inference import run_encoder_decoder_inference


parser = argparse.ArgumentParser(description="Config")
parser.add_argument("--data_path", type=str, help="Path of dataset", default="D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data\FPT_stock.xlsx")
parser.add_argument("--log_path", type=str, help="save log for training", default=r"D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\log")
parser.add_argument("--checkpoint", type=str, help="folder to save checkpoint", default=r"D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\checkpoint")
parser.add_argument("--datafile_type", type=str, help="csv, xlsx, ....", default="xlsx")
parser.add_argument("--num_rows", type=int, help="Rows of test dataset", default=720)

parser.add_argument("--batch_size", type=float, help="Batch size of dataset train, test and val", default=4)
parser.add_argument("--learning_rate", type=float, help="Learning rate for training model", default=1e-5)
parser.add_argument("--batch_first", type=bool, help="Type batch size of dataloader", default=True)
parser.add_argument("--max_viz", type=int, help="After n epochs will validation model", default=4)
parser.add_argument("--num_predicted_features", type=int, help="number of output", default=1)
parser.add_argument("--epochs", type=int, help="Epochs", default=300)
args = parser.parse_args()



if __name__ == "__main__":

    #Setting device;
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    wandb.login(key="39af6effd799f393f92bb9698e6e29404041b445")
    wandb.init(project="DSP391m-project")
    config = wandb.config
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    #Read and preprocessing dataset;
    dataset_raw = format_Dataframes(data_path=args.data_path, type_file=args.datafile_type)
    dataset = preprocessing_dataframe(dataset_raw)
    dataset = remove_outliers(dataset)
    dataset_train, dataset_test = split_data(dataset, num_rows=args.num_rows)
    dataset_train = np.array(dataset_train)
    dataset_test = np.array(dataset_test)

    #Dataloader;
    indices_data_train =get_indices_entire_sequence(data=dataset_train, window_size=35, step_size=1)
    dataloader_train = TransformerDataset(
        data=dataset_train,
        indices= indices_data_train,
        enc_seq_len= 30,
        dec_seq_len= 5,
        target_seq_len= 5,
        predict_full=False) #num_predicted_features=8 and predict_full=True if want to predict full
    train_data = DataLoader(dataset=dataloader_train, batch_size=config.batch_size)

    indices_data_test =get_indices_entire_sequence(data=dataset_test, window_size=35, step_size=1)
    dataloader_test = TransformerDataset(
        data=dataset_test,
        indices= indices_data_test,
        enc_seq_len= 30,
        dec_seq_len= 5,
        target_seq_len= 5,
        predict_full=False) #num_predicted_features=8 and predict_full=True if want to predict full
    test_data = DataLoader(dataset=dataloader_test, batch_size=config.batch_size)

    #Model
    model = TimeSeriesTransformer(
        input_size=8,
        dec_seq_len=30,
        batch_first=args.batch_first,
        num_predicted_features=args.num_predicted_features)  #num_predicted_features=8 and predict_full=True if want to predict full
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
    opt = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0, amsgrad=False)
    mse_loss = torch.nn.MSELoss() #Mean Squared Error Loss;
    mae_loss = torch.nn.L1Loss() #Mean Absolute Error Loss;
    huber_loss = torch.nn.SmoothL1Loss() #Huber Loss;
    criterion_list = [('mse', mse_loss), ("mae", mae_loss), ("huber", huber_loss)]


    #Earlt stopping;
    early_stopping = EarlyStopping(patience=10, verbose=True, path=args.checkpoint)

    #Training model;
    for epoch in range(config.epochs):

        print(f"[{epoch+1}/{args.epochs}: TRAINING PHASE]")
        loss = train_loop(model=model, datatrain=train_data, opt=opt, criterion=mse_loss, 
        epoch=epoch, path_log=args.log_path, src_mask=src_mask, tgt_mask=tgt_mask, device=DEVICE)
        print(f"[RESULT TRAINING PHASE]: MSE: {loss['MSE']}")
        wandb.log({"TRAIN LOSS": loss['MSE']})

        if (epoch+1) % args.max_viz == 0:
            print(f"[{epoch+1}/{args.epochs}: VALIDATION PHASE]")
            loss_dict = run_encoder_decoder_inference(model=model, datatrain=test_data, forecast_window=5,
            criterion_list=criterion_list, device=DEVICE)
            val_loss = loss_dict['mse']
            print(f"[RESULT VALIDATION PHASE]: MSE: {loss_dict['mse']} | MAE: {loss_dict['mae']} | HUBER: {loss_dict['huber']}")
            early_stopping(val_loss, model)
            wandb.log({"VALIDATION LOSS MSE": loss_dict['mse'], "VALIDATION LOSS MAE": loss_dict['mae'], "VALIDATION LOSS HUBER": loss_dict['huber']})

        if early_stopping.early_stop:
            print("Early stopping")
            break
    wandb.finish()






