from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Define the path to the Brave browser executable and ChromeDriver
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
chrome_driver_path = "B:/Programming/NepseBOT/chromedriver.exe"

# Set up options for using Brave
options = Options()
options.binary_location = brave_path

# Initialize the WebDriver with Brave
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)
driver.get('https://nepsealpha.com/nepse-data')
time.sleep(600)