---
title: Code Documentation Search
sdk: gradio
app_file: app.py
pinned: false
---

# Code Documentation Search

Ask questions about any public GitHub repository.
The app ingests repository files, stores embeddings in Pinecone, and answers from retrieved code context.

## Required Secrets

Set these in your Hugging Face Space Settings -> Variables and secrets:

- OPENAI_API_KEY
- PINECONE_API_KEY

## Local Run

```powershell
.\.venv\Scripts\python.exe app.py
```
