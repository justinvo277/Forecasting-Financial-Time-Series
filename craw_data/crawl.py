from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import openpyxl
import csv

#nguyenxuanquangisme

def crawl_data(num_page):

    #Crawl data on website cafef, num_page is number of pages that you want to crawl#
    #for example: crawl_data(200)#

    browser = webdriver.Chrome()  # Không cần executable_path
    browser.get('https://s.cafef.vn/lich-su-giao-dich-symbol-vnindex/trang-1-0-tab-1.chn')
    sleep(3)

    _list= []
    for i in range (0,num_page):
        data = browser.find_elements(By.XPATH, '//*[@id="render-table-owner"]/tr')
        for i in data:
            _list.append(i.text)


        new_page = browser.find_element(By.XPATH, "/html/body/form/div[3]/div[2]/div[1]/div[3]/div/div[3]/div[3]")
        new_page.click()
        sleep(3)
    return _list

#end crawl_data

def save_csv_file(list):
    #save to csv file
    # Mở tệp CSV để ghi
    with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Lặp qua từng dòng dữ liệu và ghi vào tệp CSV
        for row in list:
            # Chia dòng dữ liệu thành các trường bằng cách tách bằng khoảng trắng
            row_data = row.split()
            
            # Ghi dòng dữ liệu vào tệp CSV
            writer.writerow(row_data)

    print("Dữ liệu đã được ghi vào tệp data.csv")

#end save_csv_file

test_2_pages = crawl_data(2)
save_csv_file(test_2_pages)
