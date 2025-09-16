
import requests
_host= 'https://eu5.fusionsolar.huawei.com'
_username = 'test'
_password = 'test'

url = _host + '/thirdData/login'
headers = {
    'accept': 'application/json',
}
json = {
    'userName': _username,
    'systemCode': _password,
}
response = requests.post(url, headers=headers, json=json)
print(response.text)
_token = response.headers['xsrf-token']
print(_token)


url = _host + '/thirdData/getStationList'
json = {}
headers = {
    'Content-Type': 'application/json',
    'XSRF-TOKEN': _token,
}
response = requests.post(url, headers=headers, json=json)
print(response.text)

#----------new paginated way------------
#https://support.huawei.com/enterprise/en/doc/EDOC1100492747/c63576c8/querying-the-plant-list
url = _host + '/thirdData/stations'
json = {'pageNo':1}
headers = {
    'Content-Type': 'application/json',
    'XSRF-TOKEN': _token,
}
response = requests.post(url, headers=headers, json=json)
print(response.text)


