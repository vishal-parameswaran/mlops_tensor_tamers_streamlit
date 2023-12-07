import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os
from deeplake import VectorStore
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


@st.cache_resource
def connect_to_deeplake():
    vector_store = VectorStore(
    path = 'hub://vishalparameswaran/mlops_tensor_tamers', token=st.secrets.DEEPLAKE.token,verbose=False,overwrite=False,read_only=True,
    )
    return vector_store

def embedding_function(texts):
   return model.encode(texts)

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets/creds.json'
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # print("response")
    # print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        x = prediction.split('Output:')
        output = x[1]
        predicted = output.split('[/INST]')[0]
    return predicted


def make_prompt(question,choice):
    if choice == 'Zero Shot':
        prompt = f"""<s>[INST]You are a helpful, respectful and honest hospital assistant. Please answer the question based on the context provided. If a context is not provided, then please answer to the best of your knowledge. If you don't know the answer to a question, please don't share false information. Let's think step by step.

### Question:
{question}

### Answer:
"""
        return prompt,None
    elif choice == 'Few Shot':
        examples = [
            {"question": "How can you be smart with antibiotics?", "answer": "Only use antibiotics when prescribed by a certified healthcare provider."},
            {"question": "How should you lift objects to prevent back pain?", "answer": "Use your legs to lift, not your back. Keep the object close to your body."}
        ]
        formatted_examples = "\n\n".join([f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples])
        prompt = f"""<s>[INST]You are a helpful, respectful and honest hospital assistant. Please answer the question based on the context provided. If a context is not provided, then please answer to the best of your knowledge. If you don't know the answer to a question, please don't share false information. Use the given examples as a guide.
### Examples
{formatted_examples}
### Question:
{question}

### Answer:
"""
        return prompt,None
    elif choice == 'Chain of Thought':
        prompt = f"""<s>[INST]You are a helpful, respectful and honest hospital assistant. Please answer the question based on the context provided. If a context is not provided, then please answer to the best of your knowledge. If you don't know the answer to a question, please don't share false information. Let's think step by step.

### Question:
{question}

### Answer:
"""
        return prompt,None
    elif choice == 'RAG':
        vcs = connect_to_deeplake()
        retrieved_documents = vcs.search(embedding_data=question, embedding_function=embedding_function,k=2)
        new_context = ''
        for i in range(len(retrieved_documents['text'])):
            new_context = new_context + f'Source {i}:' + '\n' + retrieved_documents['text'][i] + '\n'
        prompt = f"""<s>[INST]You are a helpful, respectful and honest hospital assistant. Please answer the question based on the context provided. If a context is not provided, then please answer to the best of your knowledge. If you don't know the answer to a question, please don't share false information. Answer based on the contecxt provided. If you don't know the answer, please respond that you don't know.

### Context:
{new_context}

### Question:
{question}

### Answer:
"""
        return prompt,retrieved_documents

st.title('TENSOR TAMERS Base Model')

selection = st.selectbox('Select a Prompt Engineering Approach', ['Zero Shot', 'Few Shot', 'Chain of Thought', 'RAG'])

prompt = st.text_area('Enter a prompt', height=100)

if st.button('Generate'):
    prompt = make_prompt(prompt, selection)
    instance =  {
        "prompt": prompt,
        "n": 1,
        "temperature":0.1,
        "top_k":10,
        "max_tokens": 1024,
    }
    outputs,sources = predict_custom_trained_model_sample(
        project="cloud-lab-0437",
        endpoint_id="842747075387981824",
        location="us-central1",
        instances=instance
        )
    
    st.write(outputs)
    if st.button("Show Sources"):
        st.write(sources)
    