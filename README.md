# LLM Chat FastAPI Service

This project provides a simple FastAPI app that:
- reads static content from a file
- accepts a question through POST /ask
- sends the content and question to an LLM
- returns the response

## Setup

1. Create a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your API key
   ```bash
   copy .env.example .env
   ```

4. Edit .env and set:
   - OPENAI_API_KEY
   - OPENAI_MODEL (optional)
   - STATIC_CONTENT_FILE (optional, defaults to static_content.txt)

## Run

```bash
uvicorn app.main:app --reload
```

## Example request

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does this project do?\"}"
```
