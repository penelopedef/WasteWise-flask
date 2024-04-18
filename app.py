# Install required packages
# !pip install ultralytics
# !pip install opencv-python
# !pip install --upgrade  langchain langchain-community langchainhub chromadb bs4 langchain_google_genai

from flask import Flask, jsonify, request, render_template
from flask import request
import base64
import os
import json
import cv2
from ultralytics import YOLO
import pathlib

import getpass
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyB0Wm8jKfZzDRn3noTSL3Rrh3nOPV0mvkU"
os.environ["FLASK_ENV"] = "devlopment"

app = Flask(__name__)

# Load the YOLO model
model_path = 'best.pt'
object_detection_model = YOLO(model_path)

# Load chatbot model
vectorstore = Chroma(persist_directory="vectorstore_v2", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-pro")

@app.route("/")
def hello_world():

    image_path = os.path.join('test_temp', 'original.jpg')
    pred_results = get_results(image_path)
    return pred_results

@app.route("/wasteClassify", methods=['POST'])
def classifyWaste():
    f = request.form.get('image')
    #useTest = True if request.form.get("test") == "True" else False
    decoded_data = base64.b64decode(f)

    image_path = os.path.join('test_temp', "test_img.jpg")
    print(image_path)
    with open(image_path, 'wb') as f:
        f.write(decoded_data)
    #image_path = os.path.join('test_temp', 'original.jpg' if useTest is True else "test_img.jpg")
    pred_results = get_results(image_path)
    return json.dumps(pred_results)

@app.route("/chatbot", methods=['POST'])
def chat():
    question = request.form.get('user_text')

    template = """Use the following pieces of context to answer the question at the end.
    Only answer question related to the context. If it is not related to the context, answer "I can't answer that question."
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer. Dont mention about context.
    Remeber that blue bin and recyling are the same thing. In the context, garbage can also mean black bin, and organic/compost means green bin.


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

    return json.dumps(reply)



# Function to get object detection predictions
def get_pred(image_path):
    pred = object_detection_model.predict([image_path], save=True, imgsz=416, conf=0.5, iou=0.7)
    return pred

# Function to extract object detection bounding boxes
def get_boundingboxes(pred_results):
    # Iterate over each element (result) in the results list
    for result in pred_results:
        # Access the bounding boxes for the current result
        boxes = result.boxes

        # Iterate over each bounding box in the current result
        for bbox in boxes.xyxy:
            # Access the bounding box coordinates
            xmin, ymin, xmax, ymax = bbox[:4]

            # Return the bounding box coordinates
            return (xmin, ymin, xmax, ymax)

# Function to get object detection class labels and confidence scores
def get_class_and_confidence(pred_results):
    class_and_conf = []
    # Iterate over each prediction result in the list
    for result in pred_results:
        # Access the class labels for the current result
        class_labels = result.names

        # Access the bounding box coordinates, confidence scores, and class labels
        boxes = result.boxes

        # Iterate over each bounding box in the current result
        for bbox, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            # Extract the class index and confidence score
            class_index = int(cls)
            confidence_score = conf

            # Get the predicted label from the class labels
            predicted_label = class_labels[class_index]

            # Append the predicted label and confidence score to the list
            class_and_conf.append((predicted_label, confidence_score))
            
    return class_and_conf

# Function to format object detection results
def format_results(bboxes, class_and_conf):
    
    # Extract the class label and confidence score from class_and_conf list
    class_label, confidence_score = class_and_conf[0]
    
    # Extract the values from bboxes tensor
    xmin, ymin, xmax, ymax = bboxes

    # Convert the values to strings
    xmin_str = str(xmin.item())
    ymin_str = str(ymin.item())
    xmax_str = str(xmax.item())
    ymax_str = str(ymax.item())

    # Remove any unwanted characters
    class_label = class_label.strip()
    confidence_score = str(confidence_score.item())

    # Concatenate all values together with commas
    formatted_result = ','.join([xmin_str, ymin_str, xmax_str, ymax_str, class_label, confidence_score])

    return formatted_result

# Function to get object detection results
def get_results(image_path):
    pred_results = get_pred(image_path)
    
    bboxes = get_boundingboxes(pred_results)
    class_and_conf = get_class_and_confidence(pred_results)
    
    if not bboxes:
        return ',,,,No detection,'
    
    results = format_results(bboxes, class_and_conf)
    print(results)
    return results # returned as a string 'xmin_str, ymin_str, xmax_str, ymax_str, class_label, confidence_score'

# Function to format doc for chatbot
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__=="__main__":
    app.run(debug=True)   