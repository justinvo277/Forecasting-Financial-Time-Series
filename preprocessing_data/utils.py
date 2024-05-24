import os
import pandas as pd


def format_Dataframes(data_path:str=None, type_file:str="csv") -> pd.DataFrame:

    '''
    data_path: Dataset path.
    type: File type dataset, for example: data.csv, data.xlsx, ...
    '''

    if data_path is None or os.path.exists(data_path) == False:
        print("The path of dataset does not exist. Please check again !!")
    else:

        df = None
        if type_file == "csv":
            df = pd.read_csv(data_path)
        elif type_file == "xlsx":
            df = pd.read_excel(data_path)
        else:
            print("Opening this file type is not supported !!")

        column_names = ["Tên", "Ngày", 'Đóng cửa', 'Điều chỉnh', "Thay đổi", "Thay đổi 1", "%", 
        'Khối lượng (Khớp lệnh)', 'Giá trị (Khớp lệnh)', 'Khối lượng (Thỏa thuận)', 'Giá trị (Thỏa thuận)', 
        'Mở cửa', 'Cao nhất', 'Thấp nhất']


        new_column_names = df.iloc[0]
        df = df[1:]
        df.columns = new_column_names
        df.reset_index(drop=True, inplace=True)
        df.columns = column_names

        for name in df.columns:
            if name not in ["Tên", "Ngày", 'Điều chỉnh', "Thay đổi", "Thay đổi 1", "%"]:
                df[name] =  pd.to_numeric(df[name], errors='coerce')
        
        return df

        