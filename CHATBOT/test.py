import google.generativeai as genai

genai.configure(api_key="AIzaSyDSMgPLt9jaQM5adyTw7-zkTiJLFrRDdso")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("What does Inara Technologies do?")
print("✅ Response:\n", response.text)
