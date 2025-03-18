import os
import sys
import json
from openai import OpenAI

client = OpenAI()

def extract_medications(document_text):
    """
    Extracts the discharge document information as a unified JSON object.
    The returned JSON object includes:
      - "patient_information" (with sub-fields),
      - "diagnoses", "operative_finding", "operative_procedure",
      - "clinical_history", "investigation", "inpatient_management",
      - "followup", "final_remarks", and "prescribed_medications".
    
    :param document_text: The full text of the discharge document.
    :return: A JSON object (Python dict) containing the extracted information.
    """
    prompt = generate_prompt(document_text)
    result = call_openai(prompt)
    return result

def generate_prompt(document):
  
  prompt1 = f"""
    You are a medical document analysis system. Analyze the given discharge document and generate a structured JSON output as specified below.

    --- TASK INSTRUCTIONS ---
    1. FIELD PRESENCE CHECK:
      Analyze the document to determine presence of these top-level fields. For each field:
      - Return JSON object with: "present" (1/0), "remark" (brief justification)
      - Exceptions: 'patient_info' and 'clinical_history' have nested structures

      Required fields:
      - diagnoses
      - operative_finding
      - operative_procedure
      - investigation
      - inpatient_management
      - medication
      - followup

    2. PATIENT INFORMATION:
      Create nested JSON object for these sub-fields (same present/remark structure):
      {{
        "patient_number": check for medical record number,
        "name": full patient name,
        "address": physical address,
        "outcome": treatment result,
        "discharge_type": type of discharge,
        "discharge_status": current status,
        "icu_stay_length": ICU duration,
        "nshi_number": insurance ID,
        "scheme": payment method,
        "stay_duration": total hospitalization time,
        "date_of_admission": format YYYY-MM-DD,
        "date_of_discharge": format YYYY-MM-DD
      }}

    3. CLINICAL HISTORY ANALYSIS:
      Create nested JSON object checking for these specific elements. For each:
      - "present": 1 if terms found AND relevant data exists, else 0
      - "remark": specific terms found or absence note
      
      Required checks:
      {{
        "chief_complaint": ["Chief Complaints", "C/o", "C/c"],
        "history_of_presenting_illness": ["History of Present Illness", "HOPI"],
        "negative_history": ["No history of", "No h/o"],
        "past_history": ["past history", "past medical"],
        "family_history": ["family history"],
        "allergies": ["allergy", "allergen"],
        "personal_history": ["personal history"],
        "alcohol_smoking": ["alcohol use", "smoking history"],
        "comorbidities": ["hypertension", "diabetes", "T2DM", "PTB", "asthma", "COPD"],
        "vitals": ["vitals", "vital signs"],
        "examinations": ["GC", "PILCCOD", "GCS", "Chest", "Respi", "CVS", "CNS", "Per Abdomen"]
      }}

    4. MEDICATION EXTRACTION:
      - Create "prescribed_medications" array listing all medications
      - Include generic names and dosages when available

    5. FINAL REMARKS:
      - Add "final_remarks" string summarizing completeness
      - Highlight missing critical elements if any

    --- OUTPUT REQUIREMENTS ---
    - Strictly valid JSON format
    - No markdown or additional text
    - Ensure proper nesting for patient_info and clinical_history
    - Maintain original key names exactly as specified

    --- DOCUMENT INPUT ---
    {document}
    """
    
  # prompt2 = f"""
  # ==================== INSTRUCTIONS ====================

  #     You are given a medical discharge document in text format. Your task is to analyze the document and determine whether each of the following fields is present. For every field, output a JSON object with the following properties:
  #     - **Field Name**: Use the field's name as the key.
  #     - **present**: A boolean value (1 if the field is mentioned and contains relevant data, 0 if not).
  #     - **remark**: A brief note on how clearly the field is mentioned.

  #     ------------------------------------------------------
  #     FIELDS TO CHECK:
  #     ------------------------------------------------------

  #     1. **patient_information**: (Output as a nested JSON dictionary with the keys below)
  #       - patient number
  #       - name
  #       - address
  #       - outcome
  #       - discharge_type
  #       - discharge_status
  #       - icu_stay_length
  #       - nshi_number
  #       - scheme
  #       - stay_duration
  #       - date_of_admission
  #       - date_of_discharge

  #     2. **diagnoses**

  #     3. **operative_finding**

  #     4. **operative_procedure**

  #     5. **clinical_history**: (For each sub-field, create a separate JSON dictionary with keys "present" and "remark")
  #       - **chief_complaint**: Check for terms like "Chief Complaints", "C/o", "C/c".
  #       - **history_of_presenting_illness**: Check for terms like "History of Presenting Illness", "History of Present Illness", "HOPI".
  #       - **negative_history**: Check for terms like "No history of" or "No h/o".
  #       - **past_history**: Check for terms like "past" or "past history".
  #       - **family_history**: Check for terms like "family" or "family history".
  #       - **allergies**: Check for terms like "allergy" or "allergen".
  #       - **personal_history**: Check for terms like "personal" or "personal history".
  #       - **alcohol_smoking**: Check for terms like "alcohol" or "smoking".
  #       - **comorbidities**: Check for terms like "morbidities", "comorbidities", "co-morbidities" or combinations such as "hypertension" & "diabetes", "T2DM", "PTB", "asthma", "COPD".
  #       - **vitals**: Check for terms like "vitals" or "vital".
  #       - **examinations**: Check for terms such as "GC", "PILCCOD", "GCS", "Chest", "Respi", "CVS", "CNS", and "Per Abdomen" or "PA".

  #     6. **investigation**

  #     7. **inpatient_management**

  #     8. **medication**
  #       - Additionally, extract the medication information from the document. Output a JSON object with the key **prescribed_medications** and a list of medications as its value.

  #     9. **followup**

  #     ------------------------------------------------------
  #     ADDITIONAL OUTPUT:
  #     ------------------------------------------------------
  #     - Add a key named **final_remarks** at the end of your JSON output. This key should contain a string summarizing the overall findings.

  #     ------------------------------------------------------
  #     DOCUMENT:
  #     ------------------------------------------------------
  #     \"\"\" 
  #     {document} 
  #     \"\"\"

  #     ------------------------------------------------------
  #     OUTPUT FORMAT:
  #     ------------------------------------------------------
  #     Return the result as a valid JSON without any additional text.
      # """
  return prompt1

def call_openai(prompt):
    """
    Calls the OpenAI API with the provided prompt and returns the parsed JSON response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        output_text = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if output_text.startswith("```json"):
            output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
            output_text = output_text.strip()
        
        result = json.loads(output_text)
        return result
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        sys.exit(1)


