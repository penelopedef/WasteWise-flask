from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import getpass
import os

from flask import Flask, jsonify, request, render_template

os.environ["GOOGLE_API_KEY"] = "AIzaSyB0Wm8jKfZzDRn3noTSL3Rrh3nOPV0mvkU"
os.environ["FLASK_ENV"] = "devlopment"
app = Flask(__name__)

vectorstore = Chroma(persist_directory="vectorstore_v2", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# GCS bucket
# AWS S3 bucket
# Azure Blob Storage

@app.route('/')
def home():
    return render_template("index.html")


# 1. Train a global model outside/inside the flask application and save the trained model
# At inference time, you need to load the model and predict on the given sample

# Train the model with /train endpoint (save the trained model)
# At inference time, you need to load the model and predict on the given sample

@app.route("/chatbot", methods=['POST'])
def answer():

    question = request.form['ques']

    template = """Use the following pieces of context to answer the question at the end.
    Only answer question related to the context. If it is not related to the context, answer "I can't answer that question."
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer. Dont mention about context.
    Remeber that blue bin and recyling are the same thing.


    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )



    reply = rag_chain.invoke(question)
    return {
        'code': 200,
        'message': f'Replying to {question}',
        'answer': reply
    }

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

    

if __name__ == "__main__":
    app.run(debug=True)