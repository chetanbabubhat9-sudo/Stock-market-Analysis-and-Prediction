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

wait = WebDriverWait(driver, 10)

# Input start and end date
start_date = '09/03/2021'  # Format for date inputs (yyyy-mm-dd)
end_date = '09/03/2024'

# Locate the start date input field and set the date
start_date_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='date'][class='form-control btn-sm'][min='2019-09-23']")))
start_date_field.clear()
start_date_field.send_keys(start_date)

# Locate the end date input field and set the date
end_date_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='date'][class='form-control btn-sm'][max='2024-09-23']")))
end_date_field.clear()
end_date_field.send_keys(end_date)

# Ensure the fields have been filled
print("Dates set: Start Date - {}, End Date - {}".format(start_date_field.get_attribute('value'), end_date_field.get_attribute('value')))

# Add further interactions or close the browser
# driver.quit()
time.sleep(600)