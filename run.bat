@echo off
set PYTHONPATH=%cd%
uvicorn app:app --reload --host 0.0.0.0 --port 8000
