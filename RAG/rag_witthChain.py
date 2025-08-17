from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi,  TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser

# Load OPENAPI API Keys from env variable
load_dotenv()
##Indexing
# Loading the Video using Youtube API
try:
    youtubeApi = YouTubeTranscriptApi()
    transcript_list = youtubeApi.fetch("WUvTyaaNkzM")
    #converting to Corpus
    transcript = " ".join(chunk.text for chunk in transcript_list)
except:
    print("No transcript available for the Video")

# Dividing it into chunks using text splitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Convert to vectors using Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
stores = FAISS.from_documents(chunks, embeddings)

## Query (Retrivial)

retriver = stores.as_retriever(search_type="similarity", search_kwargs={"K": 4})
## Augmentation
def formatter(retrived_context):
    context_text = "\n\n".join(doc.page_content for doc in retrived_context)
    return context_text
parallelChain = RunnableParallel({
    'context': retriver | RunnableLambda(formatter),
    'question': RunnablePassthrough()
})
result = parallelChain.invoke("what is calculus")
print(result)
parser = StrOutputParser()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
     You are a helpful assistant
     Amswer ONLY from the provided context
     if the context is insufficent, just say you don't know
     
     {context}
     Question: {question}
    """,
    input_variables=['context', 'question']
)
main_chain = parallelChain | prompt | llm | parser
result = main_chain.invoke("Provide relationship of derivative with respect to integration ")
print(result)