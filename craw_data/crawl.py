import os
import csv
import pandas as pd

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By

def crawl_data(url: str, num_pages: int) -> list:
    # Crawl data on website cafef, num_page is number of pages that you want to crawl#
    browser = webdriver.Chrome()  # Không cần executable_path
    browser.get(url)
    sleep(3)
    _list = []
    for _ in range(num_pages):
        data = browser.find_elements(By.XPATH, '//*[@id="render-table-owner"]/tr')
        for i in data:
            _list.append(i.text)

        try:
            new_page = browser.find_element(By.XPATH, "/html/body/form/div[3]/div[2]/div[1]/div[3]/div/div[3]/div[3]")
            new_page.click()
            sleep(3)
        except:
            print("No more pages.")
            break
    browser.quit()
    return _list

def save_csv_file(data_list: list, save_dir: str) -> None:
    # Save to CSV file
    save_path = os.path.join(save_dir, 'stock.csv')
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data_list:
            row_data = row.split()
            writer.writerow(row_data)
    print(f"Dữ liệu đã được ghi vào tệp {save_path}")

def save_excel_file(data_list: list, save_dir: str) -> None:
    # Convert the list of strings to a DataFrame
    data = [row.split() for row in data_list]
    df = pd.DataFrame(data)
    
    # Define the save path
    save_path = os.path.join(save_dir, 'stock.xlsx')
    
    # Save the DataFrame to an Excel file
    df.to_excel(save_path, index=False, header=False)
    print(f"Dữ liệu đã được ghi vào tệp {save_path}")

