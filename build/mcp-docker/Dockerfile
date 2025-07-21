FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 3000
CMD ["python", "-m", "src.mcp.cli", "server", "--host", "0.0.0.0"]
