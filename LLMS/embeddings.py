from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import asyncio

# Load .env variables
load_dotenv()

# Debug check
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

async def main():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
    result = await embedding.aembed_query("Delhi is capital of India")
    print(result)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # macOS fix
    asyncio.run(main())


