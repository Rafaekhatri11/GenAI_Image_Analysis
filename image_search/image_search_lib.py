import os
import boto3
import json
import base64
from langchain_community.vectorstores import FAISS
from io import BytesIO

#calls Bedrock to get a vector from either an image, text, or both
def get_multimodal_vector(input_image_base64=None, input_text=None):
    
    session = boto3.Session()

    bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client
    
    request_body = {}
    
    if input_text:
        request_body["inputText"] = input_text
        
    if input_image_base64:
        request_body["inputImage"] = input_image_base64
    
    # print("request_body checking ", request_body["inputImage"])

    body = json.dumps(request_body)
    
    response = bedrock.invoke_model(
    	body=body, 
    	modelId="amazon.titan-embed-image-v1", 
    	accept="application/json", 
    	contentType="application/json"
    )
    print("checking response 2", response)
    response_body = json.loads(response.get('body').read())
    print("checking response_body 3", response_body)
    embedding = response_body.get("embedding")
    print("Checking embedding 4", embedding)
    return embedding

#creates a vector from a file
def get_vector_from_file(file_path):
    with open(file_path, "rb") as image_file:
        print("binary image file ", image_file)
        input_image_base64 = base64.b64encode(image_file.read()).decode('utf8')
        print("checking input_image_base64 1 ", input_image_base64)
    vector = get_multimodal_vector(input_image_base64 = input_image_base64)
    print("Checking vector 5", vector)
    return vector


#creates a list of (path, vector) tuples from a directory
def get_image_vectors_from_directory(path):
    items = []
    
    for file in os.listdir("images"):
        file_path = os.path.join(path,file)
        
        vector = get_vector_from_file(file_path)
        
        items.append((file_path, vector))
        
    print("checking items 6", items)
    return items


#creates and returns an in-memory vector store to be used in the application
def get_index(): 

    image_vectors = get_image_vectors_from_directory("images")
    print("checking imgae_vector 7", image_vectors)
    text_embeddings = [("", item[1]) for item in image_vectors]
    print("checking text_embedding 8", text_embeddings)
    metadatas = [{"image_path": item[0]} for item in image_vectors]
    print("Checking metadatas 9", metadatas)
    index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding = None,
        metadatas = metadatas
    )
    print("checking index 10", index)
    return index

#get a base64-encoded string from file bytes
def get_base64_from_bytes(image_bytes):
    
    image_io = BytesIO(image_bytes)
    
    print("Checking image_io 11 ",image_io)

    image_base64 = base64.b64encode(image_io.getvalue()).decode("utf-8")
    
    print("Checking image_base64 12", image_base64)

    return image_base64


#get a list of images based on the provided search term and/or search image
def get_similarity_search_results(index, search_term=None, search_image=None):
    
    search_image_base64 = (get_base64_from_bytes(search_image) if search_image else None)

    search_vector = get_multimodal_vector(input_text=search_term, input_image_base64=search_image_base64)
    
    results = index.similarity_search_by_vector(embedding=search_vector)
    
    results_images = []
    
    for res in results: #load images into list
        
        with open(res.metadata['image_path'], "rb") as f:
            img = BytesIO(f.read())
        
        results_images.append(img)
    
    
    return results_images

