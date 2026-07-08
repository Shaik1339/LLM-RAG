import logging
import os
import ssl
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

verify_ssl_global = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
if not verify_ssl_global:
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Patch the requests library which tiktoken uses under the hood
    try:
        import requests
        old_request = requests.Session.request
        def new_request(*args, **kwargs):
            kwargs['verify'] = False
            return old_request(*args, **kwargs)
        requests.Session.request = new_request
        
        # Suppress the insecure request warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="LLM File Query API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


SESSION_HISTORY: dict[str, list[dict[str, str]]] = {}


def get_content_file_path() -> Path:
    configured_path = os.getenv("STATIC_CONTENT_FILE", "static_content.txt")
    path = Path(configured_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / path).resolve()
    return path


def get_chroma_dir() -> Path:
    configured_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    path = Path(configured_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / path).resolve()
    return path


def get_embeddings() -> Any:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain.embeddings.openai import OpenAIEmbeddings
        except ImportError:
            try:
                # pyrefly: ignore [missing-import]
                from langchain_community.embeddings import OpenAIEmbeddings
            except ImportError as exc:
                raise ImportError("Could not import OpenAIEmbeddings. Please install langchain-openai.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://genailab.tcs.in/litellm/v1"
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

    embeddings_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }

    if not verify_ssl:
        embeddings_kwargs["http_client"] = httpx.Client(verify=False)

    return OpenAIEmbeddings(**embeddings_kwargs)


def get_vectorstore() -> Any:
    try:
        # pyrefly: ignore [missing-import]
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain.vectorstores import Chroma
        except ImportError:
            try:
                # pyrefly: ignore [missing-import]
                from langchain_community.vectorstores import Chroma
            except ImportError as exc:
                raise ImportError("Could not import Chroma. Please install langchain-chroma.") from exc

    embeddings = get_embeddings()
    persist_dir = get_chroma_dir()

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="documents",
    )


def sync_vector_db() -> Any:
    import hashlib

    content_path = get_content_file_path()
    if not content_path.exists():
        raise FileNotFoundError(f"Static content file not found: {content_path}")

    content = content_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Static content file is empty: {content_path}")

    persist_dir = get_chroma_dir()
    hash_file = persist_dir / "content_hash.txt"

    current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    rebuild = True
    if hash_file.exists():
        try:
            saved_hash = hash_file.read_text(encoding="utf-8").strip()
            if saved_hash == current_hash:
                rebuild = False
        except Exception as exc:
            logger.warning("Failed to read content hash: %s", exc)

    vectorstore = get_vectorstore()

    if rebuild:
        logger.info("Content changed or first run; rebuilding vector database...")
        try:
            # Try to get existing IDs and delete them to clear the database
            existing = vectorstore.get()
            if existing and "ids" in existing and existing["ids"]:
                vectorstore.delete(ids=existing["ids"])
        except Exception as exc:
            logger.warning("Failed to clear database via document IDs, dropping collection: %s", exc)
            try:
                vectorstore.delete_collection()
                # Recreate vectorstore to get a fresh collection reference
                vectorstore = get_vectorstore()
            except Exception as inner_exc:
                logger.warning("Failed to delete collection: %s", inner_exc)

        # Split content into chunks
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(content)

        # Add texts to vectorstore
        vectorstore.add_texts(texts=chunks)

        # Save the content hash
        persist_dir.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(current_hash, encoding="utf-8")
        logger.info("Vector database synchronized successfully.")

    return vectorstore


def build_prompt(question: str, content: str, history: list[dict[str, str]] | None = None) -> str:
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in history
        )
        history_text = f"\nConversation history:\n{history_text}\n"

    return f"""You are a helpful assistant. Answer the user's question using only the provided content. If the answer is not present in the content, say that you could not find it in the provided material.{history_text}

Provided content:
{content}

User question:
{question}
"""


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

    client_kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url,
    }
    if not verify_ssl:
        client_kwargs["http_client"] = httpx.Client(verify=False)

    return OpenAI(**client_kwargs)


def generate_answer_from_content(question: str, content: str, session_id: str | None = None) -> str:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    history = SESSION_HISTORY.get(session_id or "default", []) if session_id else []

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a helpful assistant. Answer strictly based on the provided content.",
                },
                {"role": "user", "content": build_prompt(question, content, history)},
            ],
        )
        return response.choices[0].message.content or ""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer strictly based on the provided content."),
            ("user", "{question}\n\nProvided content:\n{content}"),
        ]
    )
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
    llm_kwargs: dict[str, object] = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": base_url,
    }
    if not verify_ssl:
        llm_kwargs["http_client"] = httpx.Client(verify=False)
    llm = ChatOpenAI(**llm_kwargs)
    chain = prompt | llm
    result = chain.invoke({"question": question, "content": content})
    return result.content if isinstance(result.content, str) else str(result.content)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    try:
        # Sync and retrieve from the vector database
        vectorstore = sync_vector_db()
        k = int(os.getenv("RAG_K", "3"))
        docs = vectorstore.similarity_search(request.question, k=k)
        content = "\n\n".join(doc.page_content for doc in docs)
        logger.info("Retrieved relevant context from vector database for question: '%s'", request.question)
    except Exception as exc:
        logger.error("Vector database query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to query vector database: {exc}") from exc

    session_id = request.session_id or "default"
    history = SESSION_HISTORY.setdefault(session_id, [])
    history.append({"role": "user", "content": request.question})

    try:
        answer = generate_answer_from_content(request.question, content, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "source_file": str(get_content_file_path()),
        "session_id": session_id,
    }

