import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation"
)

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(token)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is capital of Denmark")

print(result.content)