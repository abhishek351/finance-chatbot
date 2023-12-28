from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI

import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = "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"
# HUGGINGFACEHUB_API_TOKEN = getpass()
# os.environ["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"] = HUGGINGFACEHUB_API_TOKEN
DB_faiss_path="vetors/db_faiss"
import requests

# API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
# headers = {"Authorization": "Bearer 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"}
prompt_template = """You are a helpful AI assistant and provide the answer to the question based on the given context. All the answers you generate must be in English because it is the language that the user understands. Answer only using context information and do not provide answers outside the given context. If the answer is not contained in the given context, just answer 'I can't answer that'.

        {context}

        Question: {question}
        Answer :"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": .2,'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    
    # llm = CTransformers(
    #     model = "llama-2-7b-chat.ggmlv3.q4_0.bin",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5
    # )
    # return llm
    OPENAI_API_BASE=""
    OPENAI_API_KEY=""
    OPENAI_API_VERSION=""
    llm= AzureChatOpenAI(deployment_name ="gpt-35-turbo",openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, openai_api_version=OPENAI_API_VERSION)

    return llm

# #QA Model 
def qa_chat():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_faiss_path, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa
#output function


# question = "? "

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# repo_id = "HuggingFaceH4/zephyr-7b-alpha "

# llm = OpenAI()
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# print(llm_chain.run(question))




def final_result(query):
    qa_result = qa_chat()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_chat()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Financial  Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    # print("yes")
    # if "I can't answer that" in answer:
    #     print("yes")
    #     sources=None
    # if sources is not None:
    #     answer += f"\nSources:" + str(sources)





    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()