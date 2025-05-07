import os
import pandas as pd
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,
            template_folder=os.path.join(r'C:\Users\91996\OneDrive\Desktop\Incident_Chatbot'))

# Configure Gemini API
genai.configure(api_key="AIzaSyDsNf8V_XDr8lLU7Tk_1e7W8BpzTQb2WIs")
model = genai.GenerativeModel("gemini-2.0-flash")  # Updated to use gemini-2.0-flash

# Initialize ChromaDB and Sentence Transformer
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="excel_docs")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Few-shot prompt examples for better accuracy
# Few-shot prompt examples for better accuracy
FEW_SHOT_EXAMPLES = [
    {"query": "For Job Name - Daily_Sales_Ingestion failed due missing input data from SalesDB, which is assigned to Data Integration Team. Provide resolution notes?",
     "answer": "Copy file to reload missing data in SalesDB and re-trigger the job."},
    {"query": "For Job Name - ReportGenerationJob failed due invalid file format in SalesReportOutput.xlsx Provide resolution notes?",
    "answer": "Copy file from source path and re-run the report generation job."},
    {"query": "For Job Name - Weekly_Inventory_Update failed due to incorrect file format. Provide resolution notes?",
     "answer": "Correct the file format and restart the job."},
    {"query": "What are the upstream and downstream dependencies for the job Daily_Sales_Ingestion?",
     "answer": "Upstream: POS Data Extraction; Downstream: BI Dashboards."},
    {"query": "What is the business impact for the job Operational_Efficiency_Metrics?",
     "answer": "Increased costs and reduced efficiency hurt profitability."},
    {"query": "Is there telemetry details related to Server crash due to high CPU usage in Server-3 on 25-03-2025",
     "answer": "On 25-03-2025, the CPU usage was 95.4%, Memory usage was 80.2% and Disk Usage was 60.5%"},
    {"query": "What is the CPU usage percentage when Server crash due to high CPU usage in Server-2 occurred on 18-03-2025",
     "answer": "On 18-03-2025, the CPU usage was 60.3%"},
    {"query": "What is the Memory usage percentage when Server crash due to high CPU usage in Server-4 occurred on 22-03-2025",
     "answer": "On 22-03-2025, Memory usage was 92.3%"},
    {"query": "What is the Disk usage percentage when Server crash due to high CPU usage in Server-5 occurred on 15-03-2025",
     "answer": "On 15-03-2025, Disk Usage was 40.8%"},
    {"query": "What could be the action or recommendation for Server crash due to high CPU usage",
     "answer": "Action - Investigate running processes consuming excessive CPU. Optimize processes and monitor the system regularly. Recommendation- Identify resource-hogging processes and either optimize them or distribute the load across additional servers. Implement alerting thresholds for CPU usage to catch high usage early."}
]

def extract_excel_content(directory):
    """Extracts text from all Excel files in a given directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                for sheet, data in df.items():
                    text = data.astype(str).values.flatten()  # Convert to string
                    content = " ".join(text)
                    documents.append((file, sheet, content))
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return documents

def index_documents(directory):
    """Indexes general document content into ChromaDB."""
    documents = extract_excel_content(directory)
    
    for i, (filename, sheet, content) in enumerate(documents):
        embedding = embed_model.encode(content).tolist()
        collection.add(documents=[content], embeddings=[embedding], ids=[f"doc-{i}"])
        print(f"Indexed document: {filename} - {sheet}")

def retrieve_document(query):
    """Finds the closest matching document content."""
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    
    if results and results.get("documents") and results["documents"][0]:
        return results["documents"][0][0]
    
    return None  # Return None if no relevant information is found

def generate_response(query):
    """First searches local documents; if no match, queries Gemini API with few-shot learning."""
    retrieved_content = retrieve_document(query)
    
    few_shot_prompt = "".join([f"Example Query: {ex['query']}\nExample Answer: {ex['answer']}\n\n" for ex in FEW_SHOT_EXAMPLES])
    
    if retrieved_content:
        prompt = f"{few_shot_prompt}User Query: {query}\n\nRelevant Information from Documents: {retrieved_content}"
    else:
        prompt = f"{few_shot_prompt}User Query: {query}\n\nNo relevant information was found in local documents. Please provide an accurate answer using external knowledge."
    
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else "Error generating response."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('user_input')  # Use .get() to avoid KeyError
    if not user_query:
        return jsonify({"error": "Missing user input"}), 400  # Return 400 Bad Request
    
    answer = generate_response(user_query)
    return jsonify({"bot_response": answer})

if __name__ == "__main__":
    excel_directory = r"C:\\Users\\91996\\OneDrive\\Desktop\\Incident_Chatbot\\prodsupp"  # Change to your directory
    index_documents(excel_directory)
    app.run(debug=True)