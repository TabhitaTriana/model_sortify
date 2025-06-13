#!/bin/bash
echo "🕒 Waiting for model load..."
sleep 2
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1