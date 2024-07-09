
import os
import torch
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataloader import TransformerDataset
from inference import run_encoder_decoder_inference
from transformer_model import TimeSeriesTransformer
from craw_data.crawl import crawl_data, save_excel_file
from utils import get_indices_entire_sequence, generate_square_subsequent_mask
from preprocessing_data.utils import format_Dataframes, preprocessing_dataframe, winsorize_dataframe



parser = argparse.ArgumentParser(description="Config")

parser.add_argument("--url", type=str, help="Link of dataset", default=None)
parser.add_argument("--num_pages", type=int, help="Number page of dataset", default=None)
parser.add_argument("--save_dir", type=str, help="Save data in your local", default=r"D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\craw_data")
parser.add_argument("--data_name", type=str, help="Data name", default=None)
parser.add_argument("--batch_first", type=bool, help="Type batch size of dataloader", default=True)
parser.add_argument("--pretrained", type=str, help="Pretrained path", default=r"D:\Major8\-DSP391m-Forecasting-Financial-Time-Series-With-Transformer\checkpoint\best.pth")
parser.add_argument("--num_predicted_features", type=int, help="Num output feauture", default=1)

args = parser.parse_args()



if __name__ == "__main__":


    #Setting device;
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
    
    print("\n")
    print(f"Crawl data form {args.url}")
    crawled_data = crawl_data(url=args.url, num_pages=args.num_pages)
    save_excel_file(data_list=crawled_data, save_dir=args.save_dir)
    print("===== The data crawling process is completed =====")
    print("Data raw")
    save_path = os.path.join(args.save_dir, 'stock.xlsx')
    df = pd.read_excel(save_path, header=None)
    df.insert(0, 'Name', 'VCB')
    df.to_excel(save_path, index=False)
    x_max = df[1].max()
    x_min = df[1].min()
    print(df)

    print("\n")
    print(f"Dataset: {args.data_name}")
    print("Data Cleaning and Preprocessing")
    df_new = format_Dataframes(data_path=save_path, data_name=args.data_name, type_file="xlsx")
    if args.data_name == "VCB":
        df_new = preprocessing_dataframe(dataFrame=df_new, data_name="VCB")
    else:
        df_new = preprocessing_dataframe(dataFrame=df_new)
    print("===== Data Cleaning and Preprocessing is completed =====")
    print(df_new)

    print("\n")
    print("Dataloader")
    '''
    Write your code here;
    '''
    dataset = winsorize_dataframe(df_new)
    dataset_test = np.array(dataset)

    indices_data_test =get_indices_entire_sequence(data=dataset_test, window_size=33, step_size=1)
    dataloader_test = TransformerDataset(
        data=dataset_test,
        indices= indices_data_test,
        enc_seq_len= 32,
        dec_seq_len= 1,
        target_seq_len= 1,
        predict_full=False) #num_predicted_features=8 and predict_full=True if want to predict full
    test_data = DataLoader(dataset=dataloader_test, batch_size=64)
    print("===== Dataloader is completed =====")

    print("\n")
    print("Model")
    model = TimeSeriesTransformer(
            input_size=4,
            dec_seq_len=32,
            batch_first=args.batch_first,
            n_decoder_layers=4,
            n_encoder_layers=4,
            num_predicted_features=args.num_predicted_features)
    model = model.float()
    model = model.to(DEVICE)
    if args.pretrained != None:
        print("Load Pretrained")
        model.load_state_dict(torch.load(args.pretrained))

    print("\n")
    print("Prediction")
    '''
    Write your code here;
    predict_thenextday = run_encoder_decoder_inference(model=model, datatrain=data, forecast_window=4, device=DEVICE)
    print(f"Giá Đóng: {predict_thenextday*(x_max - x_min) + x_min}")
    '''
    predict = run_encoder_decoder_inference(model=model, datatrain=test_data, forecast_window=1, device=DEVICE)
    print(f"Giá Đóng: {predict*(x_max - x_min) + x_min}")


    print("\n")
    print("Done !!!")