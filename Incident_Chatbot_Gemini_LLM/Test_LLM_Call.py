from google import genai

client = genai.Client(api_key="AIzaSyDsNf8V_XDr8lLU7Tk_1e7W8BpzTQb2WIs")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)