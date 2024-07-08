import torch
import wandb
import argparse
import itertools
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import TransformerDataset
from transformer_model import TimeSeriesTransformer
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe, split_data, remove_outliers, winsorize_dataframe
from utils import get_indices_entire_sequence, generate_square_subsequent_mask, train_loop, validation_loop, EarlyStopping


parser = argparse.ArgumentParser(description="Config")
parser.add_argument("--data_path", type=str, help="Path of dataset", default=r"D:\Forecasting-Financial-Time-Series\craw_data\FPT_stock.xlsx")
parser.add_argument("--log_path", type=str, help="save log for training", default=r"D:\Forecasting-Financial-Time-Series\log")
parser.add_argument("--checkpoint", type=str, help="folder to save checkpoint", default=r"D:\Forecasting-Financial-Time-Series\checkpoint")
parser.add_argument("--datafile_type", type=str, help="csv, xlsx, ....", default="xlsx")
parser.add_argument("--num_rows", type=int, help="Rows of test dataset", default=720)

parser.add_argument("--batch_size", type=float, help="Batch size of dataset train, test and val", default=128)
parser.add_argument("--learning_rate", type=float, help="Learning rate for training model", default=1e-5)
parser.add_argument("--batch_first", type=bool, help="Type batch size of dataloader", default=True)
parser.add_argument("--max_viz", type=int, help="After n epochs will validation model", default=5)
parser.add_argument("--num_predicted_features", type=int, help="number of output", default=1)
parser.add_argument("--epochs", type=int, help="Epochs", default=300)
args = parser.parse_args()



if __name__ == "__main__":

    #define grid hyperparameters
    param_grid = {
        # 'batch_size': [16, 32, 64],
        # 'learning_rate': [1e-5, 1e-4, 1e-3],
        # 'dim_val': [128, 256, 512],
        'n_stack_of_layers': [4],
        # 'n_heads': [4, 8, 12],
        # 'dropout_encoder': [0.1, 0.2, 0.3],
        # 'dropout_decoder': [0.1, 0.2, 0.3],
        # 'dim_feedforward_encoder': [512, 1024, 2048],
        # 'dim_feedforward_decoder': [512, 1024, 2048],
        # 'num_predicted_features': [1, 2, 3]
    }

    #combination of hyperparameters;
    param_combinations = list(itertools.product(*param_grid.values()))

    #Grid search;
    best_model = None
    best_loss = float('inf')
    best_params = None

    #Setting device;
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    for params in param_combinations:
        current_params = dict(zip(param_grid.keys(), params))
        print(f"Training with params: {current_params}")

        wandb.login(key="39af6effd799f393f92bb9698e6e29404041b445")
        wandb.init(project="DSP391m-project")
        config = wandb.config
        config.update(current_params)
        config.learning_rate = args.learning_rate
        config.epochs = args.epochs
        config.batch_size = args.batch_size

        #Read and preprocessing dataset;
        dataset_raw = format_Dataframes(data_path=args.data_path, type_file=args.datafile_type)
        dataset = preprocessing_dataframe(dataset_raw, fillna='ffill')
        dataset = winsorize_dataframe(dataset)
        dataset_train, dataset_test = split_data(dataset, num_rows=args.num_rows)
        dataset_train = np.array(dataset_train)
        dataset_test = np.array(dataset_test)

        #Dataloader;
        indices_data_train =get_indices_entire_sequence(data=dataset_train, window_size=32, step_size=1)
        dataloader_train = TransformerDataset(
            data=dataset_train,
            indices= indices_data_train,
            enc_seq_len= 10,
            dec_seq_len= 4,
            target_seq_len= 4,
            predict_full=False) #num_predicted_features=8 and predict_full=True if want to predict full
        train_data = DataLoader(dataset=dataloader_train, batch_size=config.batch_size)

        indices_data_test =get_indices_entire_sequence(data=dataset_test, window_size=32, step_size=1)
        dataloader_test = TransformerDataset(
            data=dataset_test,
            indices= indices_data_test,
            enc_seq_len= 10,
            dec_seq_len= 4,
            target_seq_len= 4,
            predict_full=False) #num_predicted_features=8 and predict_full=True if want to predict full
        test_data = DataLoader(dataset=dataloader_test, batch_size=config.batch_size)

        #Model
        model = TimeSeriesTransformer(
            input_size=4,
            dec_seq_len=30,
            batch_first=args.batch_first,
            n_decoder_layers=config.n_stack_of_layers,
            n_encoder_layers=config.n_stack_of_layers,
            num_predicted_features=args.num_predicted_features)  #num_predicted_features=8 and predict_full=True if want to predict full
        model = model.float()
        model = model.to(DEVICE)

        # Make src mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, enc_seq_len]
        src_mask = generate_square_subsequent_mask(dim1=4,dim2=10)
        src_mask = src_mask.to(DEVICE)

        # Make tgt mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, output_sequence_length]
        tgt_mask = generate_square_subsequent_mask(dim1=4,dim2=4)
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

            print(f"[{epoch+1}/{config.epochs}: TRAINING PHASE]")
            loss = train_loop(model=model, datatrain=train_data, opt=opt, criterion=huber_loss, 
            epoch=epoch, path_log=args.log_path, src_mask=src_mask, tgt_mask=tgt_mask, device=DEVICE)
            print(f"[RESULT TRAINING PHASE]: Loss Train: {loss['Loss Train']}")
            wandb.log({"TRAIN LOSS": loss['Loss Train']})

            if (epoch+1) % args.max_viz == 0:
                print(f"[{epoch+1}/{config.epochs}: VALIDATION PHASE]")
                loss_dict = validation_loop(model=model, datatest=test_data, criterion_list=criterion_list, src_mask=src_mask, tgt_mask=tgt_mask, device=DEVICE)
                val_loss = loss_dict['mse']
                print(f"[RESULT VALIDATION PHASE]: MSE: {loss_dict['mse']} | MAE: {loss_dict['mae']} | HUBER: {loss_dict['huber']}")
                early_stopping(val_loss, model)
                wandb.log({"VALIDATION LOSS MSE": loss_dict['mse'], "VALIDATION LOSS MAE": loss_dict['mae'], "VALIDATION LOSS HUBER": loss_dict['huber']})

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Kiểm tra nếu mô hình hiện tại là tốt nhất
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = current_params

        wandb.finish()

    print(f"Best model parameters: {best_params} with validation loss: {best_loss}")







