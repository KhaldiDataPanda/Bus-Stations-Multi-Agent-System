import requests
from bs4 import BeautifulSoup
import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def fetch_weather():
    url = "https://www.timeanddate.com/weather/@2503769"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")


    qlook = soup.find("div", id="qlook")
    temp_div = qlook.find("div", class_="h2")
    temp = temp_div.get_text(strip=True) if temp_div else "N/A"
    
    weather_type = qlook.find("p").get_text(strip=True)
    p = qlook.find_all("p")[1].get_text(strip=True)
    wind = p.split("Wind:")[1].strip() 

    
    table = soup.find("table", class_="table table--left table--inner-borders-rows")    
    humidity = None
    for row in table.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and "Humidity" in th.text and td:
            humidity = td.get_text(strip=True)
            break


    weather_data = {
        "temperature": temp,
        "weather_type": weather_type,
        "wind": wind,
        "humidity": humidity
    }
   
    #with open("weather_data.json", "w", encoding="utf-8") as f:
    #    json.dump(data, f, ensure_ascii=False, indent=4)

    return weather_data



