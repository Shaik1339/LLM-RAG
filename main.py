import logging
import os
import ssl
from pathlib import Path
from typing import Any
import base64

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Override FastAPI default error format: use 'error_message' instead of 'detail'."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Override Pydantic validation error format: use 'error_message' instead of 'detail'."""
    errors = exc.errors()
    messages = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors
    )
    return JSONResponse(
        status_code=422,
        content={"error_message": messages},
    )


from fastapi import APIRouter
from app.middlewares.guardrails import GuardrailsRoute

guardrails_router = APIRouter(route_class=GuardrailsRoute)


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


def generate_speech(text: str) -> bytes:
    """Convert text to MP3 audio using ElevenLabs API.
    
    Requires ELEVENLABS_API_KEY environment variable to be set.
    """
    import io
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed. Run: pip install requests"
        ) from exc

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
            "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
        }
    }
    
    logger.info("Generating TTS audio via ElevenLabs (voice='%s', model='%s')", voice_id, model_id)
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"ElevenLabs API request failed: {exc}") from exc


class ElevenLabsTTSRunnable(Runnable):
    """LangChain Runnable for ElevenLabs Text-to-Speech conversion."""
    
    @property
    def InputType(self):
        return str
    
    @property
    def OutputType(self):
        return dict
    
    def invoke(self, input: str, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Convert text to speech using ElevenLabs."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed") from exc
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": input,
            "model_id": model_id,
            "voice_settings": {
                "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
                "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
            }
        }
        
        logger.info("TTS Pipeline - ElevenLabs: Converting text to speech (voice='%s')", voice_id)
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            return {
                "text": input,
                "audio_base64": audio_base64,
                "audio_format": "mp3",
                "tts_provider": "elevenlabs",
                "audio_bytes": len(audio_bytes)
            }
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"ElevenLabs TTS failed: {exc}") from exc


def get_tts_runnable() -> Runnable:
    """Get the ElevenLabs TTS runnable."""
    return ElevenLabsTTSRunnable()


def build_llm_to_speech_chain(prompt_template: ChatPromptTemplate) -> Runnable:
    """Build a LangChain pipeline: DeepSeek-R1 → Text Response → TTS → Audio.
    
    Returns a runnable that chains:
    1. ChatPromptTemplate (formats the question)
    2. ChatOpenAI with DeepSeek-R1 (generates text response)
    3. TTS Runnable (converts text to audio)
    """
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    model = os.getenv("OPENAI_MODEL", "deepseek-reasoner")
    api_key = os.getenv("OPENAI_API_KEY")
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
    
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    
    # Create LLM instance
    llm_kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    if not verify_ssl:
        llm_kwargs["http_client"] = httpx.Client(verify=False)
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Get TTS runnable based on configuration
    tts = get_tts_runnable()
    
    # Extract text from LLM response and pass to TTS
    class TextExtractor(Runnable):
        @property
        def InputType(self):
            return Any
        
        @property
        def OutputType(self):
            return str
        
        def invoke(self, input: Any, config: RunnableConfig | None = None) -> str:
            if hasattr(input, 'content'):
                return input.content
            return str(input)
    
    text_extractor = TextExtractor()
    
    # Build the complete pipeline
    pipeline = prompt_template | llm | text_extractor | tts
    
    logger.info("LLM-to-Speech Pipeline initialized: DeepSeek-R1 → ElevenLabs TTS")
    
    return pipeline


def generate_answer_with_audio(question: str, content: str, session_id: str | None = None) -> dict[str, Any]:
    """Generate text answer then convert that same text to audio.

    Pipeline: LLM → text answer → ElevenLabs TTS → audio
    """
    logger.info("Executing LLM-to-Speech pipeline for question: '%s'", question)

    try:
        # Step 1: generate the text answer once
        text_answer = generate_answer_from_content(question, content, session_id=session_id)

        # Step 2: pass the generated text directly to TTS
        tts = get_tts_runnable()
        result = tts.invoke(text_answer)

        return {
            "text": text_answer,
            "audio_base64": result.get("audio_base64"),
            "audio_format": result.get("audio_format"),
            "tts_provider": result.get("tts_provider"),
            "audio_bytes": result.get("audio_bytes"),
            "status": "success",
        }
    except Exception as exc:
        logger.error("LLM-to-Speech pipeline failed: %s", exc)
        raise


def generate_answer_from_content(question: str, content: str, session_id: str | None = None) -> str:
    """Generate text answer from LLM (DeepSeek-R1) using RAG context."""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    model = os.getenv("OPENAI_MODEL", "deepseek-reasoner")  # Using DeepSeek-R1

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


@guardrails_router.post("/ask")
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
        # Use LangChain pipeline: DeepSeek-R1 → Text → TTS → Audio
        logger.info("Executing LangChain LLM-to-Speech pipeline...")
        pipeline_result = generate_answer_with_audio(request.question, content, session_id=session_id)
        answer = pipeline_result["text"]
    except Exception as exc:
        logger.error("LLM-to-Speech pipeline failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    history.append({"role": "assistant", "content": answer})

    # Build response with both text and audio from the pipeline
    response: dict[str, object] = {
        "answer": answer,
        "source_file": str(get_content_file_path()),
        "session_id": session_id,
    }
    
    # Add audio data from pipeline result if available
    if pipeline_result.get("audio_base64"):
        response["audio_base64"] = pipeline_result["audio_base64"]
        response["audio_format"] = pipeline_result.get("audio_format", "mp3")
        response["tts_provider"] = pipeline_result.get("tts_provider", "unknown")
        response["audio_bytes"] = pipeline_result.get("audio_bytes", 0)
        logger.info("LLM-to-Speech pipeline completed: %s (%d bytes) via %s",
                    response["audio_format"], response["audio_bytes"], response["tts_provider"])
    else:
        logger.warning("No audio generated from pipeline")
        response["audio_error"] = "Audio generation failed in pipeline"

    return response


app.include_router(guardrails_router)

