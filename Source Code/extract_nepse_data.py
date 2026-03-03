from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd

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

# Wait for the page to load fully
wait = WebDriverWait(driver, 100000)

# Input start and end date
start_date = '2019-09-23'  # Format for date inputs (yyyy-mm-dd)
end_date = '2024-09-23'

# Select the start date using the input field with "type=date"
start_date_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='date'][required]")))
start_date_field.clear()
start_date_field.send_keys(start_date)

# Select the end date using the input field with "type=date"
end_date_field = driver.find_elements(By.CSS_SELECTOR, "input[type='date'][required]")[1]  # Use the second date input
end_date_field.clear()
end_date_field.send_keys(end_date)

# Selecting the stock symbol from the dropdown
symbol = 'UPPER'  # Upper Tamakoshi Hydropower

# Wait for the dropdown element and click on it to open options
dropdown = wait.until(EC.element_to_be_clickable((By.NAME, 'symbol')))  # Adjust the selector as per actual HTML
dropdown.click()

# Select the desired option from the dropdown (Adjust XPATH or use a different selector if necessary)
symbol_option = driver.find_element(By.XPATH, f"//option[contains(text(), '{symbol}')]")
symbol_option.click()

# Click the Filter button
filter_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary.form-control.btn-sm[type='submit']")))
filter_button.click()

# Wait for the page to load the table after filtering
time.sleep(50000)  # Or use WebDriverWait to wait for the table to load dynamically

# Scrape the data from the table
soup = BeautifulSoup(driver.page_source, 'html.parser')
table = soup.find('table')  # Find the table element

# Parse table rows and columns to extract the data
rows = table.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    if cols:  # Avoid empty rows
        data.append(cols)

# Convert to DataFrame
columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
df = pd.DataFrame(data, columns=columns)

# Print the extracted data
print(df)

# Close the driver
# driver.quit()
