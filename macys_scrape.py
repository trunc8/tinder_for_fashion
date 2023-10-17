import requests
from bs4 import BeautifulSoup
import re
# import pandas as pd
import pickle

import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

links = []

def get_img_link(URL):
    global links
    print(URL)
    # print("..........works?")
    # URL = 'https://www.macys.com/shop/product/hanes-mens-ultimate-6pk.-crewneck-t-shirts?ID=13632373'
    driver = webdriver.Chrome()
    driver.get(URL)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    img = soup.find('picture').find('img')
    print("Name: ",img['alt'])
    print("URL: ",img['src'])
    links += [img['src']]


for index in range(5,15):
    if index == 1:
        OG_URL = 'https://www.macys.com/shop/featured/tshirts'
    else:
        OG_URL = f'https://www.macys.com/shop/featured/tshirts/Pageindex/{index}'
    
    OG_driver = webdriver.Chrome()
    OG_driver.get(OG_URL)


    OG_html = OG_driver.page_source
    OG_soup = BeautifulSoup(OG_html, 'html.parser')

    articles = OG_soup.find_all('div', class_="productDescription smaller-margin")

    i=1
    for article in articles:
        print(i)
        article_name = article.find('a')
        get_img_link("https://www.macys.com/"+article_name['href'])
        # break
        i=i+1


    with open(f'img_urls_{index}.pkl', 'wb') as f:
        pickle.dump(links, f)


