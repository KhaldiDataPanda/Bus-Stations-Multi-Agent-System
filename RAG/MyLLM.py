from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from RAG.scraper import fetch_weather


def LLM_speed_reduction_estimation(passanger_counts ,Think= False):
    """This function invokes the weather scraper nad uses teh weather data and passangercounts
    to estimate speed reduction using a language model."""

    weather_data = fetch_weather()

    if Think == True:
        model_n = "qwen3:1.7b"
    
    else :
        model_n = "gemma3:1b"


    system_prompt = """  You are given a task to estimate speed reduction for Toyota Coaster bus operating between inside a dense city.
        your role is to quickly estemate the percentage of speed reduction based on the weather conditions and number of passangers.
        
        Notes :       
        - Number of passangers at medium is around 25-40 , High loads are around 45-75 passangers, below 20 passangers is considered low load.         
        - In Nice Wearther Condtions :Low Passanger load don't reduce speed. For medium passangers load, the reduction can start from 1 up to 4 .  and starts from around 7 up to 15 percentages  for larger loads. 
        - High/Medium loads reduce up to 1.5 to 2 more speed while being in high temperature/humidity condtions.
        - For special weather conditions (like rain, wind, snow, fog...) the reduction starts from 5 up to 15 percentages (based on the siverity of the condition), and can exceed that to reach range of 20 up to 30  for highly harsh conditions.          

        Output Format:
        Reduction: <integer Value>
        - The output must be only the integer value without the % symbol
        - Don't follow up with any additional text or symbols.

        """
        

    user_template = """Based on the given data:
    Weather data: {weather_data} 
    Passenger count: {passanger_counts}    
    Calculate the speed reduction for this bus using the above rules. 
    """


    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_template)
    ])

    model = OllamaLLM(model=model_n)

    chain = prompt | model

    response = chain.invoke({
        "weather_data": weather_data,
        "passanger_counts": passanger_counts
    })


    return int(response.strip().split(":")[-1]) 
    


if __name__ == "__main__":
    print(LLM_speed_reduction_estimation(55, True))
    print(LLM_speed_reduction_estimation(55))

    print(LLM_speed_reduction_estimation(30, True))
    print(LLM_speed_reduction_estimation(30))
    
    print(LLM_speed_reduction_estimation(12, True))
    print(LLM_speed_reduction_estimation(12))




