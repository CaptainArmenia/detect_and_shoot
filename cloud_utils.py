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


# Notify machine's heartbeat to cloud
def notify_heartbeat():
    headers = {"x-api-key": api_key}
    endpoint = "machines/notify-heartbeat"
    response = requests.post(url + endpoint, headers=headers)
    print(f"Heartbeat notification result: {response}")


# Notify that the machine went to sleep
def notify_hibernation():
    headers = {"x-api-key": api_key}
    endpoint = "machines/notify-hibernation"
    response = requests.post(url + endpoint, headers=headers)
    print(f"Hibernation notification result: {response}")