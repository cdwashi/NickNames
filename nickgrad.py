import os
#import openai
import gradio as gr
import config

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

llm = ChatOpenAI(temperature=0.9)

#Here I'm using a generic prompt template instead of assigning it explicitly to the system prompt
prompt1 = ChatPromptTemplate.from_template(
    "What is a funny nickname to describe \
    a person who does or makes {product}?"
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

#Here I've created a clear and specific system prompt template for the LLM's system message instead of just being specific in a general template
template = """You are a witty creative writer with the task of helping \
    clients come up with humourous nicknames for the jobs they have or their hobby. \
    Your job is to write in 25 words or less a witty story for how the following \
    person got his nickname. Pay strict attention to the gender \
    of the person in the story and check that you use the correct pronoun in the story."""
human_template = "{product}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chain2 = LLMChain(llm=llm, prompt=chat_prompt)

overall_simp_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

#overall_simp_chain.run(user_product)

def nickname_generator(product):
    output = overall_simp_chain.run(product)
    
    return output

testgr = gr.Interface(fn=nickname_generator, 
                      inputs=[gr.Textbox(label="Product Made or Job Held by Person the Nickname is For")], 
                      outputs=[gr.Textbox(label="Witty Nickname and Origin Story")],
                      title="🤖  NICKNAMER😊: The Nickname Generator! by IST Group - AI. Powered by OpenAI.  🤖",                      
                      description="Enter a product or job and get a witty nickname and origin story! 🙄",
                      examples=[["dog walker"], ["a person who makes miniature cars"], ["a person who fixes broken computers"]],
                      allow_flagging="never"
                      )
testgr.launch(share=True)

gr.close_all()
