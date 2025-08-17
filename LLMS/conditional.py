from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of feedback")


pyndaticParser = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template="Classify the sentiment of feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables={'feedback'},
    partial_variables={"format_instruction": pyndaticParser.get_format_instructions()}

)



classifier_Chain = prompt1 | model | pyndaticParser
# result = classifier_Chain.invoke({'feedback':  'This is a very bad jersey'}).content
# print(result)

posPrompt = PromptTemplate(
    template="Write an appropriate response to positive feedback \n {feedback}",
    input_variables=['feedback']
)

negPrompt = PromptTemplate(
    template="Write an appropriate response to negative feedback \n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', posPrompt | model | parser),
    (lambda x:x.sentiment=='negative', negPrompt | model | parser),
    lambda x: "could not find sentiment"
)

chain = classifier_Chain | branch_chain

result = chain.invoke({'feedback':  'This is a very bad jersey'})
print(result)