from azure.ai.documentintelligence.document_analysis_client import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
import json


# Read all keys from environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_docintelligence_key = os.getenv('AZURE_DOCINTELLIGENCE_KEY')
azure_docintelligence_endpoint = os.getenv('AZURE_DOCINTELLIGENCE_ENDPOINT')
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME","gpt-4.1-nano")

def get_pdf_data(fpath):
    """
    Extract text from PDF using Azure Document Intelligence (Layout model).
    """
    if not azure_docintelligence_key or not azure_docintelligence_endpoint:
        raise ValueError("Azure Document Intelligence credentials not set in config.yaml")

    client = DocumentAnalysisClient(
        endpoint=azure_docintelligence_endpoint,
        credential=AzureKeyCredential(azure_docintelligence_key)
    )

    with open(fpath, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", document=f)
        result = poller.result()

    text = ""
    for page in result.pages:
        for line in page.lines:
            text += line.content + "\n"
    return text

def get_llm():
      # Define the AzureOpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview"
    )

    return client

def get_invoice_info_from_llm(data):
    llm = get_llm()
    prompt = "Act as an expert in extracting information from medical invoices. You are given with the invoice details of a patient. Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data. Return the data in json format = {'disease':"",'expense':""}"
    messages=[
        {"role": "system", 
        "content": prompt}
        ]
    
    user_content = f"INVOICE DETAILS: {data}"

    messages.append({"role": "user", "content": user_content})

    response = llm.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.4,
                max_tokens=2500)
        
    data = json.loads(response.choices[0].message.content)

    return data

if __name__ == '__main__':
    bill_folder = "Bills"
    bill_name = "MedicalBill1.pdf"

    bill_path = os.path.join(bill_folder, bill_name)
    if not os.path.exists(bill_path):
        print(f"{bill_path} does not exist. Please check the file location")
    else:
        bill_info = get_pdf_data(bill_path)
        invoice_details = get_invoice_info_from_llm(bill_info)

    print(f"Disease: {invoice_details['disease']}")
    print(f"Expense: {invoice_details['expense']}")
