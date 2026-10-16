import urllib.request
import re

url = "https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color/Cherry_(including_sour)___healthy"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(r'"path":"raw/color/[^"]+/([^"]+\.(?:JPG|jpg|jpeg|png))"', html)
        print("Found image matches in HTML:", len(matches))
        if matches:
            print("Sample match:", matches[:3])
except Exception as e:
    print("Error:", e)
