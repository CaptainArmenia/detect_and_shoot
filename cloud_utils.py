import requests

api_key = "Hwf8YYVcG0qnb5lZOBFflw"
url = "https://sneakycameras.com/"

def report_activity(file):
    file_dict = {'file': open(file,'rb')}
    headers = {"x-api-key": api_key}
    endpoint = "report-activity/"
    response = requests.post(url + endpoint, headers=headers, files=file_dict)
    print(f"Upload result: {response}")