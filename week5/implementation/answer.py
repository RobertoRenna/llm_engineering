from pathlib import Path
import os
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Use HuggingFace embeddings (same as day2/day3)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Configure OpenAI client for Databricks or OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    MODEL = "gpt-4o-mini"  # Fast and cost-effective OpenAI model
    openai = OpenAI()
else:
    # Use Databricks AI Gateway
    MODEL = "databricks-gpt-oss-120b"  # Databricks free tier model
    try:
        # dbutils is a global object in Databricks notebooks, not a module
        databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    except:
        databricks_token = os.environ.get("DATABRICKS_TOKEN", "dummy-token")
    
    openai = OpenAI(
        api_key=databricks_token,
        base_url="https://7474647277163805.ai-gateway.cloud.databricks.com/mlflow/v1"
    )

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    
    # Build messages for OpenAI API
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    # Call OpenAI API (works with both OpenAI and Databricks)
    response = openai.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    content = response.choices[0].message.content
    
    # Handle Databricks structured response format if needed
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                answer = item.get('text', '')
                break
        else:
            answer = str(content)
    else:
        answer = content
    
    return answer, docs
