# Import necessary libraries
import os, re
from flask import Flask, render_template, request
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import io
from azure.storage.blob import BlobServiceClient


from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from azure.ai.documentintelligence import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import LLMChain
import json
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# Get the OpenAI API key from the environment variable

api_key= os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME","gpt-4.1-nano")
embbedings_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME","text-embedding-3-small")
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME","bupa-docs")

if api_key is None or api_key == "":
    print("OpenAI API key not set or empty. Please set the environment variable.")
    exit()  # Terminate the program if the API key is not set.

# Initialize the OpenAI client with the API key
os.environ['AZURE_OPENAI_API_KEY'] = api_key
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]




# Update document loader to optionally use blob storage
def get_document_loader(use_blob=True, blob_connection_string=BLOB_CONNECTION_STRING, container_name=CONTAINER_NAME):
    if use_blob and blob_connection_string and container_name:
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        docs = []
        for blob in container_client.list_blobs():
            if blob.name.lower().endswith(".pdf"):
                blob_client = container_client.get_blob_client(blob)
                pdf_bytes = blob_client.download_blob().readall()
                pdf_file = io.BytesIO(pdf_bytes)
                loader = PyPDFLoader(pdf_file)
                docs.extend(loader.load())
        return docs
    else:
        loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        docs = loader.load()
        return docs

def get_text_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)

    embeddings= AzureOpenAIEmbeddings(
        model=embbedings_deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview"
    )
    db = FAISS.from_documents(
        chunks, embedding=embeddings
    )
    return db


def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever

def get_claim_approval_context():
    db = get_embeddings()
    context = db.similarity_search("What are the documents required for claim approval?")
    claim_approval_context = ""
    for x in context:
        claim_approval_context += x.page_content

    return claim_approval_context

def get_general_exclusion_context():
    db = get_embeddings()
    context = db.similarity_search("Give a list of all general exclusions")
    general_exclusion_context = ""
    for x in context:
        general_exclusion_context += x.page_content

    return general_exclusion_context

def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        # Use Azure Document Intelligence to extract text
        azure_docintelligence_key = os.getenv("AZURE_DOCINTELLIGENCE_KEY")
        azure_docintelligence_endpoint = os.getenv("AZURE_DOCINTELLIGENCE_ENDPOINT")
        if not azure_docintelligence_key or not azure_docintelligence_endpoint:
            raise ValueError("Azure Document Intelligence credentials not set in environment variables.")

        client = DocumentAnalysisClient(
            endpoint=azure_docintelligence_endpoint,
            credential=AzureKeyCredential(azure_docintelligence_key)
        )
        # file.stream is a file-like object for Flask uploads
        poller = client.begin_analyze_document("prebuilt-layout", document=file.stream)
        result = poller.result()
        for page in result.pages:
            for line in page.lines:
                text += line.content + "\n"
    return text

def get_bill_info(data):
    prompt = "Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {'disease':"",'expense':""}"
    messages=[
        {"role": "system", 
        "content": prompt}
        ]
    
    user_content = f"INVOICE DETAILS: {data}"

    messages.append({"role": "user", "content": user_content})

     # Define the AzureOpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview"
    )

    response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.4,
                max_tokens=2500)
        
    data = json.loads(response.choices[0].message.content)

    return data



PROMPT = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. Use the following guidelines for your analysis.

1.Verify if the patient has provided all necessary information and all necessary documents
and if you find any incomplete information or required documents are not provided then set INFORMATION criteria as FALSE and REJECT the claim.
if patient has provided all required documents then set INFORMATION criteria as TRUE. 

2. If any disease mentioned in the medical bill of the patient is in the general exclusions list, set EXCLUSION criteria as FALSE and REJECT the claim.

Use this information to verify if the application is valid and to accept or reject the application.

DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}
EXCLUSION LIST : {general_exclusion_context}
PATIENT INFO : {patient_info}
MEDICAL BILL : {medical_bill_info}

Use the above information to verify if the application is valid and decide if the application has to be accepted or rejected keeping the guidelines into consideration. 

Generate a detailed report about the claim and procedures you followed for accepting or rejecting the claim and the write the information you used for creating the report. 
Create a report in the following format

Write whether INFORMATION AND EXCLUSION are TRUE or FALSE 
Reject the claim if any of them is FALSE.
Write whether claim is accepted or not. If the claim has been accepted, the maximum amount which can be approved will be {max_amount}

Executive Summary
[Provide a Summary of the report.]

Introduction
[Write a paragraph about the aim of this report, and the state of the approval.]

Claim Details
[Provide details about the submitted claim]

Claim Description
[Write a short description about claim]

Document Verification
[Mentions which documents are submitted and if they are verified.] 

Document Summary
[Give a summary of everything here including the medical reports of the patient]

Please verify for any signs of fraud in the submitted claim if you find the documents required for accepting the claim for the medical treatment.
"""


prompt = PromptTemplate(input_variables=["claim_approval_context", "general_exclusion_context", "patient_info","max_amount"], template=PROMPT)

def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template, threshold=0.4):
    vectorizer = CountVectorizer()
    patient_info_vector = vectorizer.fit_transform([claim_reason])

    for disease in general_exclusion_list:
        disease_vector = vectorizer.transform([disease])
        similarity = cosine_similarity(patient_info_vector, disease_vector)[0][0]
        if float(similarity) > float(threshold):
            
            prompt_template = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and give a good rejection. You the following guidelines for your analysis.
            PATIENT INFO : {patient_info}

            Executive Summary
                [Provide a Summary of the report.]

                Introduction
                [Write a paragraph about the aim of this report, and the state of the approval.]

                Claim Details
                [Provide details about the submitted claim]

                Claim Description
                [Write a short description about claim]

                Document Verification
                [Mentions which documents are submitted and if they are verified.] 

                Document Summary
                [Give a summary of everything here including the medical reports of the patient]
            
            CLAIM MUST BE REJECTED: Patient has {disease} which is present in the general exclusion list."""
            return prompt_template
    
    return prompt_template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def msg():
    claim_validation_message = ""
    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']
    medical_bill = request.files['medical_bill']
    total_claim_amount = request.form['total_claim_amount']
    description = request.form['description']

    bill = get_file_content(medical_bill)

    bill_info = get_bill_info(bill)
    # If input amount is more than the bill amount - REJECT
    if bill_info['expense'] != None and int(bill_info['expense']) < int(total_claim_amount) :
        claim_validation_message = "The amount mentioned for claiming is more than the billed amount. Claim Rejected."
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=claim_validation_message)
        
    elif bill_info['expense'] != None and int(bill_info['expense']) > int(total_claim_amount) :
        #Check if the disease is in the exclusion list or not, update the prompt accordingly
        patient_info = f"Name: {name} " + f"\nAddress: {address} " + f"\nClaim type: {claim_type} " + f"\nClaim reason: {claim_reason}" + f"\nMedical facility: {medical_facility} " + f"\nDate : {date} " + f"\nTotal claim amount: {total_claim_amount}" + f"\nDescription: {description}"
        medical_bill_info = f"Medical Bill: {bill}"
        
        validated_prompt = check_claim_rejection(bill_info["disease"], general_exclusion_list,PROMPT)
    
        prompt_template = PromptTemplate(input_variables=["claim_approval_context","patient_info"],template=validated_prompt)
        llm = AzureOpenAI(model=deployment_name)
        llmchain = LLMChain(llm=llm, prompt= prompt_template)
        output = llmchain.run({"claim_approval_context": get_claim_approval_context(), "general_exclusion_context": get_general_exclusion_context(), "patient_info": patient_info, "medical_bill_info":medical_bill_info,"max_amount":total_claim_amount, "disease":bill_info["disease"]})
        
        output = re.sub(r'\n', '<br>', output)
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)
        
    else:
        #If no expense value has been extracted
        output = "Please enter a valid Consultation Receipt."
        
        return render_template("result.html", name=name, address=address, claim_type=claim_type, claim_reason=claim_reason, date=date, medical_facility=medical_facility, total_claim_amount=total_claim_amount, description=description, output=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)

