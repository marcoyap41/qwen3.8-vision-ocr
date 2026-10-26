# Qwen 3.8 Vision OCR
### Experimental AI Engineering Project: OCR Application based on Groq Qwen 3.8 27B, with a FastAPI backend and a React (Vite) frontend

## Install Dependency
```powershell
# Backend dependencies (run in project root)
uv sync

# Frontend dependencies
cd frontend
npm install
```

## Setup API Key
```env
# Add your Groq API Key inside .env file (project root)
GROQ_API_KEY=your_groq_api_key_here
```

## How to run app
```powershell
# Terminal 1 - Backend (project root) -> http://127.0.0.1:8000
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2 - Frontend -> http://localhost:5173
cd frontend
npm run dev
```


> marcoyap41
