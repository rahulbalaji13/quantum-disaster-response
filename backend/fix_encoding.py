import requests

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    r = requests.get(url)
    if r.status_code == 200:
        with open("imagenet_classes.txt", "w", encoding="utf-8") as f:
            f.write(r.text)
        print("Successfully downloaded and saved imagenet_classes.txt as UTF-8.")
    else:
        print(f"Failed to download. Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
