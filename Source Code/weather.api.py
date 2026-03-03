import requests
import datetime
import csv

# Replace with your actual WeatherAPI key
API_KEY = '0e8f6131a57b43f482035215243112'
city = 'Kathmandu'

# Get today's date and calculate the date 60 days ago
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=60)

# Function to format the date as YYYY-MM-DD
def format_date(date):
    return date.strftime("%Y-%m-%d")

# Define CSV file path
csv_file_path = 'historical_weather_data_kathmandu_60_days.csv'

# Create or open the CSV file and write the header
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Date', 'Humidity (%)', 'Wind Speed (mph)', 'Precipitation (mm)', 'PM2.5 (µg/m³)', 'AQI'
    ])  # Header row

    # Loop through all the dates from 60 days ago to today
    current_date = start_date
    while current_date <= end_date:
        formatted_date = format_date(current_date)
        url = f"http://api.weatherapi.com/v1/history.xml?key={API_KEY}&q={city}&dt=2010-01-01"

        # Send the request to the WeatherAPI
        response = requests.get(url)

        # Debug: Print the status code and response
        print(f"Status Code for {formatted_date}: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            try:
                # Extract the required data
                humidity = data['forecast']['forecastday'][0]['day']['avghumidity']
                wind_speed = data['forecast']['forecastday'][0]['day']['maxwind_mph']
                precipitation = data['forecast']['forecastday'][0]['day']['totalprecip_mm']
                pm25 = data['forecast']['forecastday'][0]['hour'][0]['air_quality']['pm2_5']
                aqi = data['forecast']['forecastday'][0]['hour'][0]['air_quality']['us-epa-index']

                # Write the extracted data as a row in the CSV
                writer.writerow([
                    formatted_date, humidity, wind_speed, precipitation, pm25, aqi
                ])

                print(f"Data for {formatted_date} written to CSV.")
            except KeyError:
                print(f"Data for {formatted_date} is incomplete or missing.")
        else:
            print(f"Failed to retrieve data for {formatted_date}. Response: {response.text}")

        # Move to the next day
        current_date += datetime.timedelta(days=1)

print(f"Data collection completed. The CSV file is saved as '{csv_file_path}'.")
