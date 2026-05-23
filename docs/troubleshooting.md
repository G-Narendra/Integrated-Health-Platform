# Troubleshooting Guide - Integrated Healthcare Platform

## Common Issues & Solutions

### API Key Issues

**Problem:** "GEMINI_API_KEY not configured" error
**Solution:**
1. Create `.env` file with `GEMINI_API_KEY=your_key_here`
2. Or enter API key in the Streamlit sidebar
3. Or set environment variable: `export GEMINI_API_KEY=your_key`
4. Verify key is valid at https://aistudio.google.com/app/apikey

**Problem:** "API Error: API_KEY_INVALID"
**Solution:**
1. Check key hasn't expired
2. Regenerate key in Google AI Studio
3. Ensure billing is enabled if using paid tier

### ChromaDB Issues

**Problem:** "ChromaDB init error"
**Solution:**
```bash
# Clear and rebuild ChromaDB
rm -rf data/chroma_db
python scripts/build_vector_db.py
```

**Problem:** "Collection not found"
**Solution:**
```bash
# Reinitialize all collections
python scripts/ingest_documents.py
```

### Streamlit Issues

**Problem:** "Address already in use" on port 8501
**Solution:**
```bash
# Find and kill process
netstat -ano | findstr :8501
taskkill /PID <PID> /F
# Or use different port
streamlit run app.py --server.port 8502
```

**Problem:** App crashes on startup
**Solution:**
1. Check Python version: `python --version` (needs 3.10+)
2. Verify all deps installed: `pip install -r requirements.txt`
3. Check for missing imports in app.py

### SQLite Issues

**Problem:** "database is locked"
**Solution:**
1. Ensure only one app instance is running
2. Restart the application
3. If persists: `rm data/audit.db` and restart

### LLM Response Issues

**Problem:** Empty or nonsensical responses
**Solution:**
1. Check API key quota usage
2. Simplify the prompt (reduce context length)
3. Reduce temperature to 0.1-0.3 for more deterministic output
4. Check internet connectivity

**Problem:** Response timeout
**Solution:**
1. Reduce max_tokens parameter
2. Simplify the medical case input
3. Check Gemini API status page

### Package Issues

**Problem:** ModuleNotFoundError
**Solution:**
```bash
# Ensure using correct virtual environment
which python  # Check path includes ../venv/

# Reinstall packages
pip install --upgrade -r requirements.txt
```

### Performance Issues

**Problem:** Slow responses
**Solution:**
1. Check internet speed (Gemini API requires connectivity)
2. Reduce ChromaDB query results (n_results=3 instead of 5)
3. Reduce input text length
4. Consider upgrading Gemini API tier

### Vector Database Issues

**Problem:** Empty search results
**Solution:**
```bash
# Rebuild vector database with medical knowledge
python scripts/build_vector_db.py
# Verify database exists
ls -la data/chroma_db/
```

## Health Check

Verify system is operational:
```bash
# Check all components
curl http://localhost:8501/_stcore/health
# OR for FastAPI
curl http://localhost:8000/api/v1/health
```

## Logs

```bash
# Application logs
# Streamlit creates logs in .streamlit/logs/

# Docker logs
docker-compose logs -f app

# Access audit logs
sqlite3 data/audit.db "SELECT * FROM system_requests ORDER BY id DESC LIMIT 10;"
```

## Getting Help

- Check [Gemini API docs](https://ai.google.dev/docs)
- Raise issues on GitHub repository
- Contact UAE MOH IT support for compliance questions
