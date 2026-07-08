# LLM Chat App Specification

## 1. Overview
This application is a small FastAPI service that reads content from a static text file, accepts a user question, sends the question and content to an LLM-compatible API, and returns the generated answer.

## 2. Goals
- Provide a simple HTTP API for answering questions from static content.
- Read content from a configurable text file.
- Use environment-based configuration for API key, model, and base URL.
- Return plain JSON responses for easy testing.

## 3. Functional Requirements

### 3.1 Health Check
- Endpoint: GET /health
- Returns a JSON response indicating the service is running.

### 3.2 Ask Endpoint
- Endpoint: POST /ask
- Request body:
  {
    "question": "string"
  }
- Behavior:
  - Load content from the configured static file.
  - Build a prompt combining the content and the user question.
  - Send the prompt to the configured LLM provider.
  - Return the generated response as JSON.

### 3.3 Static Content Source
- The application reads content from a file specified by the STATIC_CONTENT_FILE environment variable.
- If not provided, it defaults to static_content.txt.

### 3.4 LLM Configuration
The app uses the following environment variables:
- OPENAI_API_KEY: API key for the LLM provider.
- OPENAI_MODEL: model name to use.
- OPENAI_BASE_URL: base URL for the OpenAI-compatible endpoint.
- OPENAI_VERIFY_SSL: optional SSL verification toggle.

## 4. Non-Functional Requirements
- Simple and easy to run locally.
- Works with FastAPI and Uvicorn.
- Uses environment variables for configuration.
- Supports local development and testing.

## 5. Request/Response Flow

### 5.1 Ask Flow
1. Client sends POST /ask with JSON containing a question.
2. Server loads static content from the configured file.
3. Server builds a prompt.
4. Server sends the prompt to the configured LLM endpoint.
5. Server returns the answer in JSON.

Example response:
{
  "answer": "...",
  "source_file": "..."
}

## 6. Technical Stack
- Python
- FastAPI
- Uvicorn
- OpenAI Python SDK
- dotenv
- httpx

## 7. Deployment Notes
- Run the app with Uvicorn.
- Expose the service on port 4000 for local testing.
- Ensure the .env file contains valid LLM configuration values.

## 8. Future Enhancements
- Add authentication for the API.
- Support multiple static content sources.
- Add logging and monitoring.
- Add automated tests for the API endpoints.
