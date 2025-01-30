# Traffic Data **Australian Government data**
This dataset is provided by the [South Australian Government Data Directory](https://data.sa.gov.au/data/dataset/traffic-intersection-count) in `.csv` format

Data sourced from the traffic signal system (SCATS). The dataset contains the number of cars passing through a given intersection each hour within the Adelaide City Council jurisdiction, *related to 122 intersections*.

Traffic volume estimation is calculated every hour and the intersection location is identified by latitude and longitude, and the type of intersection is also present.
## Dataset Features
After searching online, the data schema used is for a **traffic intersection count** dataset.
- *site_no*: This is a unique identifier assigned to each intersection, enabling easy reference and differentiation.
- *site_desc*: This provides a brief description of the intersection, which could include details like street names, location landmarks, etc.
- *lat*: This represents the latitude coordinate of the intersection.
- *lon*: This represents the longitude coordinate of the intersection.
- *rec_date*: This indicates the date on which the traffic count was conducted at the intersection.
- *rec_hour*: This specifies the hour within the day for which the traffic count was recorded.
- *hourly_traffic*: This attribute holds the actual number of vehicles that passed through the intersection during the specified hour.
- *t_detector_counts*: This represents the total expected number of vehicles the system anticipated for the specific hour at the intersection, potentially based on historical data or other factors.
- *t_valid_counts*: This attribute indicates the total number of valid traffic counts recorded during the hour, excluding any potential errors or discrepancies.
- *t_error_counts*: This specifies the total number of identified errors or inconsistencies in the traffic count data for the specific hour.
- *error_ratio*: This is the calculated ratio between the number of errors and the number of valid counts, offering insight into the data quality and potential inaccuracies.

> 2010-01-01 to 2014-12-31 every hour

| site_no | site_desc                      | lat          | lon          | rec_date   | rec_hour | hourly_traffic | t_detector_counts | t_valid_counts | t_error_counts | error_ratio |
| ------- | ------------------------------ | ------------ | ------------ | ---------- | -------- | -------------- | ----------------- | -------------- | -------------- | ----------- |
| 3009    | SIGNALISED 4 WAY INTERSECTION  | -34.93292601 | 138.600274   | 2014-09-16 | 15:00:00 | 1700.0         | 288               | 288.0          | 0.0            | 0.0         |
| 3155    | SIGNALISED PEDESTRIAN CROSSING | -34.93341632 | 138.59028769 | 2011-03-05 | 22:00:00 | 111.0          | 288               | 288.0          | 0.0            | 0.0         |
| 3022    | SIGNALISED 4 WAY INTERSECTION  | -34.92686101 | 138.60588557 | 2014-07-20 | 16:00:00 | 2014.0         | 288               | 288.0          | 0.0            | 0.0         |
| 3045    | SIGNALISED 4 WAY INTERSECTION  | -34.93627172 | 138.58817817 | 2013-08-25 | 01:00:00 | 990.0          | 288               | 288.0          | 0.0            | 0.0         |
| 3053    | SIGNALISED 4 WAY INTERSECTION  | -34.91556988 | 138.59886491 | 2013-01-21 | 15:00:00 | 2707.0         | 288               | 288.0          | 0.0            | 0.0         |
| 3051    | SIGNALISED 4 WAY INTERSECTION  | -34.91195768 | 138.59880765 | 2014-12-03 | 01:00:00 | 155.0          | 288               | 288.0          | 0.0            | 0.0         |


> [!Warning] The file weighs more than 500MB and has over 5,369,323 rows!
> When using it, it will certainly be advisable to reorganize the data and reduce the number of intersections taken into account.
# Map data acquisition

There is *open Street Map* which provides data free of charge in `.osm` format

# Weather API: Open-meteo

Provides all data free of charge, both historical and future, with less than 10,000 calls per day

> They have their own Python library with guides on their [website](https://open-meteo.com/en/docs/historical-weather-api)

Example of historical data:

> [!NOTE]
> In the data request, you can specify which data you want; many more can be specified.

| time             | temperature_2m (°C) | precipitation (mm) | rain (mm) | snowfall (cm) |
| ---------------- | ------------------- | ------------------ | --------- | ------------- |
| 2024-02-22T10:00 | 8.2                 | 0.40               | 0.40      | 0.00          |
| 2024-02-22T11:00 | 8.4                 | 0.70               | 0.70      | 0.00          |
| 2024-02-22T12:00 | 8.7                 | 0.40               | 0.40      | 0.00          |
| 2024-02-22T13:00 | 8.3                 | 0.70               | 0.70      | 0.00          |
| 2024-02-22T14:00 | 8.5                 | 0.50               | 0.50      | 0.00          |
| 2024-02-22T15:00 | 8.2                 | 0.100              | 0.50      | 0.00          |