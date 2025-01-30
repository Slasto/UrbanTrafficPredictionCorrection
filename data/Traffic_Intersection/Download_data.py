import requests
import zipfile
import io
import os

def extract_csv_files(zip_file):
    FolderPath = os.path.dirname(os.path.abspath(__file__))
    for file_info in zip_file.infolist():
        if file_info.filename.endswith('.csv'):
            zip_file.extract(file_info.filename, path=FolderPath)

def main() :
    # src https://data.sa.gov.au/data/dataset/traffic-intersection-count
    url = "https://s3.ap-southeast-2.amazonaws.com/dmzweb.adelaidecitycouncil.com/OpenData/Traffic+Intersection/intersection_data.zip"

    print("Downloading 'intersection_data.zip' ...")
    response = requests.get(url)

    print("Unzip 'intersection_data.csv' ...")
    zip = zipfile.ZipFile(io.BytesIO(response.content))

    extract_csv_files(zip)
    print("Done!")

    return 1

if __name__ == "__main__" :
	main()