import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Tool use to get the current time
def get_time():
    return datetime.now().strftime("%H:%M:%S")

# Tool use to get the weather for a city
def get_weather(city):
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY not found in environment variables"}
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # Use metric units (Celsius)
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}


# Initialize OpenAI client with API key from environment
if not OPENAI_API_KEY: 
    print({"error": "OPENWEATHER_API_KEY not found in environment variables"})
    quit()
client = OpenAI(api_key=OPENAI_API_KEY)


def toolful_answer(prompt, model="gpt-5-mini"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions and help with tasks."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function", 
                "function": {
                    "name": "get_time", 
                    "description": "Access the current time", 
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_weather", 
                    "description": "Access the weather for any given city", 
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to get the weather for"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
    )

    tool_calls = response.choices[0].message.tool_calls
    answer = response.choices[0].message.content
    if tool_calls: 
        messages.append({"role": "assistant", "content": answer, "tool_calls": tool_calls})
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            if tool_name == "get_weather":
                weather = get_weather(tool_args.get("city", None))
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "name": tool_name,
                    "content": json.dumps(weather)
                })
            elif tool_name == "get_time":
                time = get_time()
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "name": tool_name,
                    "content": json.dumps(time)
                })

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        answer = response.choices[0].message.content

    return answer


def toolless_answer(prompt, model="gpt-5-mini"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions and help with tasks."},
        {"role": "user", "content": prompt}
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages, 
        stream=True
    )
    return stream


if __name__ == "__main__":
    prompt = "Hello! How is the weather today in Berlin? Also, what day is it going to be in 5 days?"
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    response = toolful_answer(prompt)
    toolful_time = time.time() - start_time
    print(f"Toolful Response: {response}")
    print(f"Toolful Time: {toolful_time:.2f} seconds")
    
    print("\n" + "="*50 + "\n")

    prompt = "Hello! How is the weather today in Tokyo?"
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    response = toolless_answer(prompt)
    toolless_time = time.time() - start_time
    print(f"Toolless Response:")
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)
    print(f"\nToolless Time: {toolless_time:.2f} seconds")
