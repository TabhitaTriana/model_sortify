#!/bin/bash

# Fallback ke 8000 jika $PORT tidak diset
PORT=${PORT:-8000}

echo "🕒 Waiting for model load..."
sleep 2
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
