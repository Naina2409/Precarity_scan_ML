FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/generate_data.py && python scripts/train.py --no-tune
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
