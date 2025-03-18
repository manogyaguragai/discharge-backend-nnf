import os
import json
import pdfplumber
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from llm1_medications import extract_medications
from llm2_dosage_check import check_medications_dosage
from openai import OpenAI

# Ensure OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Error: OPENAI_API_KEY environment variable is not set.")
OpenAI.api_key = openai_api_key

# Initialize FastAPI app
app = FastAPI()

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    try:
        with pdfplumber.open(pdf_file.file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

@app.post("/process_pdf/")
async def process_pdf(pdf: UploadFile = File(...)):
    """Process an uploaded PDF file, calling LLM1 and LLM2, and return results."""
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
    
    # Extract text
    document_text = extract_text_from_pdf(pdf)
    
    # Call LLM1 to extract medications
    llm1_result = extract_medications(document_text)
    prescribed_medications = llm1_result.get("prescribed_medications", [])
    
    if not prescribed_medications:
        return {"message": "No medications found."}
    
    # Call LLM2 to check dosages
    final_output = check_medications_dosage(prescribed_medications)
    
    # Format results
    return {
        "file_name": pdf.filename,
        "extracted_medications": llm1_result,
        "dosage_check_results": final_output
    }
