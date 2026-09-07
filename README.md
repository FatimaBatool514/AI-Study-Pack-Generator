# AI Study Pack Generator — Multi-Stage AI Workflow

## Architecture

PDF -> Extraction -> Chunking -> Embeddings -> FAISS

Then:

1. Planning Agent
2. Retrieval / Context Agent
3. Content Generation Agent
4. Assessment Agent
5. Review Agent
6. Refinement Agent
7. Final Study Pack

The stages share a `WorkflowContext` object.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GROQ_API_KEY` as an environment variable or enter it in the Streamlit sidebar.

## Deploy

Push `app.py` and `requirements.txt` to GitHub and deploy the repository using Streamlit Community Cloud.

For deployment, store the API key in Streamlit Secrets rather than committing it to GitHub.

Example secret:

```toml
GROQ_API_KEY = "your-key"
```

## Notes

- FAISS is in-memory for this starter project.
- The application currently supports text-based PDFs.
- OCR can be added for scanned PDFs.
- For production multi-user use, persist indexes and isolate document data by user/session.
