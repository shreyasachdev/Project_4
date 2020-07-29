from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import random
import time
import math
import urllib.request



class scrape_listing():
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_driver = os.getcwd() +"\\chromedriver.exe"
        self.driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver)
        self.base_url = "https://www.kijiji.ca/b-cars-trucks/canada/new__used/c174l0a49"
    
    def page_num(self):
        self.driver.get(self.base_url)
        time.sleep(5)
        try:
            total_ads_unrefined  = self.driver.find_element_by_css_selector('.titlecount').text
            print(total_ads_unrefined)
            total_ads = total_ads_unrefined.replace('(', '').replace(')', '').replace(',','')
            
            total_page_num =  math.ceil(int(total_ads) / 40)

            print(total_page_num)

        except:
            total_page_num = 1
            pass    
        return total_page_num

    def scrape_ad(self):
        self.driver.get(f"https://www.kijiji.ca/b-cars-trucks/canada/new__used/page-{self.Random_number(100)}/c174l0a49")
        time.sleep(5)
        ad_links = (self.driver.find_elements_by_css_selector("div[class*='search-item']"))
        time.sleep(5)
        listing = random.choice(ad_links)
        Individal_url = "https:/www.kijiji.ca"+ str(listing.get_attribute("data-vip-url"))
        self.driver.get(Individal_url)
        time.sleep(5)
        image_src = self.driver.find_element_by_css_selector("img[itemprop='image']").get_attribute("src")
        image_model = self.driver.find_element_by_css_selector("dd[itemprop='brand']").text
        self.Image_save(image_src)
        return {'image_score': image_src, 'image_model': image_model}

    def Random_number(self, max_number):
        return random.randint(1, max_number)

    def Image_save(self, image_src):
        urllib.request.urlretrieve(image_src, "IMAGE.png")


    
