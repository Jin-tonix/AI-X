import requests

API_URL = "https://api-inference.huggingface.co/models/google-t5/t5-base"
headers = {"Authorization": "Bearer hf_***"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Меня зовут Вольфганг и я живу в Берлине",
})