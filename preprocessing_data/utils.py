import os
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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


def preprocessing_dataframe(dataFrame: pd.DataFrame, fillna: str="mean", scale: str=None) -> pd.DataFrame:
    '''
    dataFrame: A data frame is data after reading from a csv file and having run it through the format_Dataframes() function.
    fillna: Type of fill data NaN, Null or None; [None, Zero, Mean, Linear, Ffill].
    scale: Type of scale; [MinMaxScaler, StandardScaler]
    '''

    # Drop unnecessary columns
    dataFrame.drop(columns=['Điều chỉnh', 'Thay đổi', 'Thay đổi 1', '%', 'Khối lượng (Khớp lệnh)',
                            "Giá trị (Khớp lệnh)", "Khối lượng (Thỏa thuận)", "Giá trị (Thỏa thuận)"], inplace=True)
    # Replace "NaN" strings with np.nan
    dataFrame.replace("NaN", np.nan, inplace=True)
    
    # Handle NaN values
    if fillna == "zero":
        float_columns = dataFrame.select_dtypes(include=['float']).columns
        dataFrame[float_columns] = dataFrame[float_columns].fillna(0)
        int_columns = dataFrame.select_dtypes(include=['int']).columns
        dataFrame[int_columns] = dataFrame[int_columns].fillna(0)
    elif fillna == "mean":
        float_columns = dataFrame.select_dtypes(include=['float']).columns
        dataFrame[float_columns] = dataFrame[float_columns].fillna(dataFrame[float_columns].mean())
        int_columns = dataFrame.select_dtypes(include=['int']).columns
        dataFrame[int_columns] = dataFrame[int_columns].fillna(dataFrame[int_columns].mean())
    elif fillna == 'linear':
        dataFrame = dataFrame.interpolate(method='linear')
    elif fillna == 'ffill':
        dataFrame = dataFrame.ffill()
    else:
        dataFrame.dropna(inplace=True)

    dataFrame = dataFrame.bfill()
    
    # Separate 'Ngày' and 'Tên' columns
    tmp_dataFrame_day = dataFrame["Ngày"]
    tmp_dataFrame_day.reset_index(drop=True, inplace=True)
    tmp_dataFrame_name = dataFrame["Tên"]
    tmp_dataFrame_name.reset_index(drop=True, inplace=True)
    dataFrame.drop(columns=["Ngày", "Tên"], inplace=True)
    dataFrame.reset_index(drop=True, inplace=True)

    # Scale the data
    if scale == "std":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    tmp_scaler = scaler.fit_transform(dataFrame)
    dataFrame = pd.DataFrame(tmp_scaler, columns=dataFrame.columns)
    
    # Concatenate 'Ngày' column back to the DataFrame
    dataFrame = pd.concat([tmp_dataFrame_day, dataFrame], axis=1)
    dataFrame.set_index("Ngày", inplace=True)
    dataFrame.sort_values(by="Ngày", inplace=True)
    
    return dataFrame


def split_data(df: pd.DataFrame, num_rows: int) -> tuple:
    df_train = df.iloc[:-num_rows]
    df_test =  df.iloc[-num_rows:]
    return df_train, df_test


def remove_outliers(df: pd.DataFrame, columns: list=['Đóng cửa', 'Khối lượng (Khớp lệnh)', 'Giá trị (Khớp lệnh)', 
    'Khối lượng (Thỏa thuận)', 'Giá trị (Thỏa thuận)', 'Mở cửa', 'Cao nhất', 'Thấp nhất']):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        df = df.loc[filter]
    return df


def winsorize_series(series, limits):
    """
    Thực hiện Winsorization trên một cột dữ liệu.
    
    Parameters:
    series (pandas.Series): Cột dữ liệu cần Winsorization.
    limits (tuple): Tuple chứa tỷ lệ Winsorization ở hai đầu (lower limit, upper limit).
    
    Returns:
    pandas.Series: Cột dữ liệu sau khi Winsorization.
    """
    return pd.Series(winsorize(series, limits=limits), index=series.index)

def winsorize_dataframe(df, limits=(0.075, 0.075)):
    """
    Thực hiện Winsorization trên toàn bộ DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame chứa dữ liệu cần Winsorization.
    limits (tuple): Tuple chứa tỷ lệ Winsorization ở hai đầu (lower limit, upper limit). Mặc định là (0.05, 0.05).
    
    Returns:
    pandas.DataFrame: DataFrame sau khi Winsorization.
    """
    df_winsorized = df.copy()
    for col in df_winsorized.columns:
        if pd.api.types.is_numeric_dtype(df_winsorized[col]):
            df_winsorized[col] = winsorize_series(df_winsorized[col], limits)
    return df_winsorized


def create_sliding_windows(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)