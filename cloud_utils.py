import requests
import json


secrets_file = ".secrets/secrets.json"
secrets = json.load(open(secrets_file))

api_key = secrets.get("api_key")
url = secrets.get("url")

# Report detected activity to cloud by sending a clip
def report_activity(file):
    file_dict = {'file': open(file,'rb')}
    headers = {"x-api-key": api_key}
    endpoint = "report-activity"
    response = requests.post(url + endpoint, headers=headers, files=file_dict)
    print(f"Upload result: {response}")

# Send heartbeat to cloud
def send_heartbeat():
    headers = {"x-api-key": api_key}
    endpoint = "heartbeat"
    response = requests.post(url + endpoint, headers=headers)
    print(f"Heartbeat result: {response}")