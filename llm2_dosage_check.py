import os
import sys
import json
import Stemmer

from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LLMOpenAI

# Define the persist directory for the BNF vector store and the BNF PDF path
PERSIST_DIR = "./nnf_storage"
NNF_PDF_PATH = "/home/manogyaguragai/Desktop/Projects/discharge-doc/nnf.pdf"

def initialize_nnf_bm25(nnf_pdf_path=NNF_PDF_PATH):
    """
    Initialize or load a BM25 retriever using a persisted NNF vector store.
    If the vector store does not exist, create it from the provided NNF PDF and persist it.
    """
    Settings.num_workers = 8
    Settings.llm = LLMOpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            print("Loaded persisted NNF vector store.")
        except Exception as e:
            print(f"Error loading persisted storage: {e}")
            sys.exit(1)
    else:
        try:
            reader = SimpleDirectoryReader(
                input_files=[nnf_pdf_path],
                file_metadata=lambda x: {"filename": x},
            )
            documents = reader.load_data()
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
            nodes = node_parser.get_nodes_from_documents(documents)
            
            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)
            storage_context.persist(persist_dir=PERSIST_DIR)
            print("Created and persisted new NNF vector store.")
        except Exception as e:
            print(f"Error creating NNF vector store: {e}")
            sys.exit(1)
    
    docstore = storage_context.docstore
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=5,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    return bm25_retriever

def cross_reference_dosage(medication, bm25_retriever):
    """
    For a given medication (with keys "name" and "dosage"), query the BM25 retriever
    to fetch relevant NNF nodes and use an LLM to extract the recommended dosage.
    
    The returned JSON object includes:
      - name
      - prescribed_dosage (dosage from above)
      - nnf_dosage
      - correctness (1 if the prescribed dosage exactly matches the NNF dosage, otherwise 0)
      - remarks (a brief explanation)
    """
    # Construct prescribed_dosage from available fields
    prescribed_dosage = f"{medication.get('dosage', '')}".strip()
    
    # Initial query with brand name
    query = f"{medication['name']} dosage"
    retrieved_nodes = bm25_retriever.retrieve(query)

    # Concatenate text from the retrieved nodes
    context = "\n".join([node.get_text() for node in retrieved_nodes])
    print("butter")
    print(context)
    
    prompt = f"""
        You are a medical assistant specialized in pharmacology.
        Given the medication details below and the NNF reference information, determine the recommended dosage.

        Medication:
        - Name: {medication['name']}
        - Prescribed dosage: {prescribed_dosage}

        NNF Reference Information:
        \"\"\"
        {context}
        \"\"\"

        From the NNF reference, extract the recommended dosage for the medication and compare it with the prescribed dosage.
        Return a JSON object with the following keys:
        - name: medication name
        - prescribed_dosage: as provided above
        - nnf_dosage: the dosage extracted from the NNF reference
        - correctness: 1 if the prescribed dosage exactly matches the NNF dosage, otherwise 0
        - remarks: a brief note explaining the comparison, if the nnf dosage is not found, mention that the given medicine name is brand.

        Return only the JSON object.
        """
    try:
        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        output_text = response.choices[0].message.content.strip()
        if output_text.startswith("```json"):
            output_text = output_text[7:-3].strip()
        result = json.loads(output_text)
        # Ensure the brand_name flag is correctly set in the result
        return result
    except Exception as e:
        print(f"Error in cross_reference_dosage for {medication['name']}: {e}")
        sys.exit(1)

def check_medications_dosage(medications):
    """
    For each medication in the provided list, use the persisted BM25 retriever to cross reference the dosage.
    Returns a JSON object with key "medications_analysis" containing a list of analysis results.
    """
    bm25_retriever = initialize_nnf_bm25()
    final_results = []
    for medication in medications:
        analysis = cross_reference_dosage(medication, bm25_retriever)
        final_results.append(analysis)
    return {"medications_analysis": final_results}
