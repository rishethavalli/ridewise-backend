# Hugging Face Spaces entry point
# This file imports and exposes the FastAPI app from main.py

from main import app

# The 'app' variable must be present at module level for Hugging Face Spaces to find it
# Hugging Face will automatically run: uvicorn app:app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
