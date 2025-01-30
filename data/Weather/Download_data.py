'''
$ pip install openmeteo-requests requests-cache retry-requests numpy pandas
'''
import os

import openmeteo_requests
import requests_cache
import pandas as pd

from retry_requests import retry
from openmeteo_sdk import WeatherApiResponse

def WheatearApiResponse_to_DataFrame(response : WeatherApiResponse, requiredData : list[str]) -> pd.DataFrame:
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for index, name in zip(range(len(requiredData)), requiredData):
        hourly_data[name] = hourly.Variables(index).ValuesAsNumpy()
    return pd.DataFrame(data=hourly_data)

def main():
	requiredData = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "wind_speed_10m", "cloud_cover"]

	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": -34.9287,
		"longitude": 138.5986,
		"start_date": "2010-01-01",
		"end_date": "2014-12-31",
		"hourly": requiredData,
	} # DOC: https://open-meteo.com/en/docs/historical-weather-api
	Cache_file = '.cache'
	CSVFile = os.path.dirname(os.path.abspath(__file__))+'/data.csv'

	# Setup delle Open-Meteo API con cache e ritentativi in caso di errore
	cache_session = requests_cache.CachedSession(Cache_file, expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openMeteo = openmeteo_requests.Client(session = retry_session)

	print("Invio della richiesta...")
	try:
		response = openMeteo.weather_api(url, params=params)[0]
	except Exception as Err_Response:
		print(f"Errore: {Err_Response}")
		return -1
	del openMeteo
	print("Nessun errore nella ricezione\nInformazioni generali:")
	print(f"- Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"- Elevation {response.Elevation()} m asl")
	print(f"- Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	print(f"- Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
	
	print(f"Salvataggio dati ricevuti...")
	WheatearApiResponse_to_DataFrame(response, requiredData).to_csv(CSVFile, index = True)
	
	print(f"Eliminazione file di cache...")
	os.remove(Cache_file+'.sqlite')
	print("Finito!")
	return 1

if __name__ == "__main__" :
	main()