import requests

url = 'http://localhost:8080/api'

r = requests.post(url, json = {"X" : ["I hate Brokeback Mountain"]})

print(r.json())