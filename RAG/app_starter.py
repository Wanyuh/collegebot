import json
import os
from datetime import datetime
import tiktoken
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncpg
import hashlib
import uvicorn
import boto3
from botocore.exceptions import ClientError
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, RedirectResponse
import fitz  # PyMuPDF
from markdownify import markdownify as md
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from fastapi import FastAPI, Request
import random
from openai import OpenAI
import logging

app = FastAPI()

# Mount the static files directory to serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

AWS_ACCESS_KEY_ID = 'id'
AWS_SECRET_ACCESS_KEY = 'pwd'
S3_BUCKET_NAME = 'rds-pdf-files'
S3_REGION = 'us-west-2'

# PostgreSQL database configuration
DB_USER = 'postgres'
DB_PASS = 'thisisatest'
DB_NAME = 'postgres'
DB_HOST = 'rag-project.cr482uc82ad7.us-west-2.rds.amazonaws.com'
DB_PORT = '5432'

OPENAI_API_KEY = "key"

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
QA_PROMPT_PATH = os.path.join(current_file_dir, 'prompt', 'qa_prompt.txt')

async def get_db_connection():
    conn = await asyncpg.connect(
        user=DB_USER, password=DB_PASS, database=DB_NAME, host=DB_HOST, port=DB_PORT
    )
    return conn

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)

# Function to calculate MD5 hash of the file
def calculate_md5(file: UploadFile):
    md5_hash = hashlib.md5()
    for chunk in iter(lambda: file.file.read(4096), b""):
        md5_hash.update(chunk)
    file.file.seek(0)  # Reset file pointer after reading
    return md5_hash.hexdigest()


async def upload_to_s3(key: str, body: bytes, content_type: str):
    """Upload the given content to S3."""
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType=content_type
    )


async def generate_random_id(conn):
    while True:
        # Generate a random integer between 1 and 10^9 (or any range you prefer)
        random_id = random.randint(1, 10 ** 9)

        # Check if this ID already exists in the database
        existing_id = await conn.fetchrow("SELECT 1 FROM max_kb_file WHERE id = $1", random_id)
        if existing_id is None:
            return random_id

async def save_file_info_to_db(conn, file_hash, file, file_content, target_name):
    """Save the uploaded file info to PostgreSQL."""
    file_id = hashlib.md5(file.filename.encode('utf-8')).hexdigest()
    create_time = datetime.utcnow()
    update_time = datetime.utcnow()

    id = await generate_random_id(conn)


    await conn.execute(
        """
        INSERT INTO max_kb_file (
            id, md5, filename, file_size, user_id, platform, region_name,
            bucket_name, file_id, target_name, create_time, update_time
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        )
        """,
        id,
        file_hash,  # md5 hash
        file.filename[:64],
        len(file_content),
        "user123",
        "web",
        S3_REGION,
        S3_BUCKET_NAME,
        file_id[:64],
        target_name[:64],
        create_time,
        update_time
    )
    return file_id

async def store_parsed_document_info(conn, parsed_file_name, parsed_content, file_id):
    """Store the parsed document information in the database."""
    files_info = json.dumps({
        "file_id": file_id,
        "filename": parsed_file_name,
        "target_name": f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/parsed/{parsed_file_name}"
    })


    id = await generate_random_id(conn)
    await conn.execute(
        """
        INSERT INTO max_kb_document (
            id, name, char_length, status, is_active, type, meta, dataset_id,
            hit_handling_method, directly_return_similarity, files, creator, create_time
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11::json, $12, $13
        )
        """,
        id,
        parsed_file_name,
        len(parsed_content),
        "uploaded",  # status
        True,  # is_active
        "markdown",
        "{}",  # meta as an empty JSONB object
        3,  # dataset_id
        "none",  # hit_handling_method
        0.0,  # directly_return_similarity
        files_info,
        "user123",
        datetime.utcnow()  # Create time
    )

async def process_parsed_content(conn, parsed_content, file_id):
    """Split and process the parsed content into paragraphs and vectors."""
    paragraphs = split_markdown_to_paragraphs(parsed_content)
    bert_vectorizer = BertVectorizer()

    id_paragraph = await generate_random_id(conn)

    id_paragraph_document = await generate_random_id(conn)

    for index, paragraph in enumerate(paragraphs):
        # Insert the paragraph into the max_kb_paragraph table
        paragraph_id = await conn.fetchval(
            """
            INSERT INTO max_kb_paragraph (
                id, content, title, status, hit_num, is_active, dataset_id, document_id, 
                creator, create_time, updater, update_time, tenant_id
            ) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) 
            RETURNING id
            """,
            id_paragraph,
            paragraph,  # content
            f"Paragraph {index + 1}",  # title (or any title logic you want)
            "active",  # status (or any status logic you want)
            0,  # hit_num
            True,  # is_active
            4,  # dataset_id
            id_paragraph_document,  # document_id
            "user123",  # creator (example)
            datetime.utcnow(),  # create_time
            "user123",  # updater (example)
            datetime.utcnow(),  # update_time
            0  # tenant_id
        )

        # Generate vector for the paragraph
        vector = bert_vectorizer.generate_vector(paragraph)
        # Prepare the vector as a space-separated string
        vector_array = '[' + ','.join(map(str, vector.tolist())) + ']'

        # Prepare the meta information (can be modified as needed)
        meta_info = {
            "additional_info": "Any other relevant metadata"
        }
        meta_info_json = json.dumps(meta_info)

        id_vector = await generate_random_id(conn)
        id_vector_document = await generate_random_id(conn)
        # Insert the vector into the 'max_kb_embedding' table
        # This table will support qa function
        await conn.execute(
            """
            INSERT INTO max_kb_embedding (
                id, source_id, source_type, is_active, embedding, meta, dataset_id, 
                document_id, paragraph_id, search_vector, creator, create_time, updater, update_time, tenant_id
            ) VALUES (
                $1, $2, $3, $4, $5::VECTOR, $6::JSONB, $7, $8, $9, to_tsvector($10), $11, $12, $13, $14, $15
            )
            """,
            id_vector,
            id_paragraph,
            "paragraph",
            True,  # is_active
            vector_array,
            meta_info_json,  # JSONB field
            5,
            id_vector_document,
            paragraph_id,
            paragraph,  # Using paragraph text for search_vector generation (optional)
            "user123",  # Creator (example)
            datetime.utcnow(),  # Create time
            "user123",  # Updater (example)
            datetime.utcnow(),  # Update time
            0  # Tenant ID
        )


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()

def parse_pdf_to_markdown(pdf_bytes: bytes) -> str:
    """Parse PDF file into Markdown format."""
    # Open the PDF file
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    markdown_content = ""

    # Iterate through the pages and extract text
    for page in doc:
        markdown_content += md(page.get_text())

    return markdown_content

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure the file is not empty
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Initialize the hash object
        hasher = hashlib.md5()
        file_content = bytearray()

        # Read the file in chunks and update the hash
        chunk_size = 8192  # 8 KB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            file_content.extend(chunk)

            file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty or unreadable")

        # Get the hash of the entire file content
        file_hash = hasher.hexdigest()

        # S3 key for the original file
        s3_key = f"uploads/{file.filename}"
        target_name = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"

        # Database connection
        conn = await get_db_connection()

        try:
            # Check if a file with the same hash exists
            existing_file = await conn.fetchrow("SELECT * FROM max_kb_file WHERE md5 = $1", file_hash)
            if existing_file:
                raise HTTPException(status_code=400, detail="File already exists")

            # Upload the file to S3
            await upload_to_s3(s3_key, file_content, file.content_type)

            # Save file info to 'max_kb_file'
            file_id = await save_file_info_to_db(conn, file_hash, file, file_content, target_name)

            # Parse the PDF file into Markdown format
            parsed_content = parse_pdf_to_markdown(bytes(file_content))
            parsed_file_name = f"{file.filename}.md"
            parsed_file_key = f"parsed/{parsed_file_name}"

            # Upload the parsed Markdown file to S3
            await upload_to_s3(parsed_file_key, parsed_content.encode('utf-8'), 'text/markdown')

            # Store the parsed document information in the 'max_kb_document'
            await store_parsed_document_info(conn, parsed_file_name, parsed_content, file_id)

            # Process the parsed content to 'max_kb_paragraph' and Insert the vector into the 'max_kb_embedding' table
            await process_parsed_content(conn, parsed_content, file_id)

        finally:
            await conn.close()

        # Return success response
        return RedirectResponse(url="/qa_index", status_code=303)

    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/qa_index", response_class=HTMLResponse)
async def qa_index():
    with open("static/qa_index.html") as f:
        return f.read()


def split_markdown_to_paragraphs(markdown_text):
    # Split the markdown text by double newlines to get paragraphs
    return [paragraph.strip() for paragraph in markdown_text.split("\n\n") if paragraph.strip()]

# Dummy function to generate vectors (replace with your actual vector generation logic)
class BertVectorizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set the model to evaluation mode

    def generate_vector(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.model(**inputs)

        # Use the embeddings from the last hidden state (mean pooling)
        # Here we take the mean of the token embeddings to get a single vector for the input text
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return vector



def calculate_similarity(vec1, vec2):
    # Ensure both vectors are 1D NumPy arrays
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Input vectors must be 1D arrays")

    # Example: Cosine similarity
    cosine_similarity =  np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return cosine_similarity


@app.post("/handle_user_question")
async def handle_user_question(request: Request):
    # Step 1: Get the JSON data from the request
    data = await request.json()  # Use await to get the JSON body
    user_question = data.get('question')

    # Step 2: Vectorize the user question
    bert_vectorizer = BertVectorizer()
    question_vector = bert_vectorizer.generate_vector(user_question)

    # Step 3: Fetch embeddings from the database
    conn = await get_db_connection()  # Get a database connection
    query = """
            SELECT e.source_id, e.embedding, p.content
            FROM max_kb_embedding e
            JOIN max_kb_paragraph p ON p.id = e.source_id;
        """
    results = await conn.fetch(query)
    await conn.close()

    # Step 4: Calculate similarities with the vectors fetched from the database
    similarities = []
    for row in results:
        id = row["source_id"]
        vector = row["embedding"]
        content = row["content"]

        # If the vector is stored as a list or array, you might need to adjust this
        if isinstance(vector, str):
            # Convert the string representation to bytes
            float_list = list(map(float, vector.strip('[]').split(',')))  # Assuming it's a comma-separated string
            vector = np.array(float_list, dtype=np.float32)
        elif isinstance(vector, (list, tuple)):
            vector = np.array(vector, dtype=np.float32)  # Convert to NumPy array
        else:
            vector = np.frombuffer(vector, dtype=np.float32)  # This assumes it's already a byte array


        sim_score = calculate_similarity(question_vector, vector)
        # similarities[row["id"]] = sim_score
        similarities.append({
            "similarity": sim_score,
            "id": id,
            "content": content
        })


    # Step 5: Sort and filter the fragments by similarity score
    # Get the top N fragments
    N = 3
    top_fragments = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:N]

    response_data = {
        "question": user_question,
        "results": [
            {
                "similarity": float(frag["similarity"]),
                # "content": frag["content"],
                "answer": generate_from_gpt(frag["content"], user_question)
            }
            for frag in top_fragments
        ],
    }


    return response_data

def read_file(file_path):
    """ Helper function to read a file content. """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def truncate_to_token_limit(text, max_tokens, model):
    """Truncate text to fit within the specified token limit."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Truncate tokens to the limit
    return encoding.decode(tokens)



def generate_from_gpt(background_info, question):
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt_template = read_file(QA_PROMPT_PATH)

    formatted_prompt = system_prompt_template.format(background=background_info, question=question)

    max_allowed_tokens = 4096 - 400
    truncated_prompt = truncate_to_token_limit(formatted_prompt, max_allowed_tokens, "gpt-4o")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant. Given the following background information, provide a detailed answer to the question"},
            {"role": "user", "content": truncated_prompt}
        ],
        seed=42,
        temperature=0.3,
        max_tokens=400,
    )
    res = response.choices[0].message.content
    return res




if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)


