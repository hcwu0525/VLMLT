from openai import OpenAI

client = OpenAI(
    base_url="https://api.datapipe.app/v1",
    api_key="sk-0zMtaFA6orOE29d042561d2b9bE5407bA52151605b28Ea54"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! Can you translate this text into French: Guten tag! Wie geht's dir?"},
  ]
)

print(completion)