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

#user_product = input("What is the product you want to describe? ")
#system_Prompt = "You are a witty creative writer with the task of helping clients come up with humourous nicknames for the jobs they have or their hobby. 

prompt1 = ChatPromptTemplate.from_template(
    "What is a funny nickname to describe \
    a person who does or makes {product}?"
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

prompt2 = ChatPromptTemplate.from_template(
    "Write in 25 words or less a witty story for how the following \
    person got his nickname:{nick_name}"
)

prompt3 = ChatPromptTemplate.from_template(
    "Write in 25 words or less a witty story for how the following \
    person got his nickname. Pay strict attention to the gender \
    of the person in the story and check that you use the correct pronoun in the story:{nick_name}"
)

chain2 = LLMChain(llm=llm, prompt=prompt3)

overall_simp_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

#overall_simp_chain.run(user_product)

def nickname_generator(user_product):
    output = overall_simp_chain.run(user_product)
    
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
