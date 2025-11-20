# frontend/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install  -r requirements.txt

COPY front.py .

EXPOSE 7860

CMD ["python", "front.py"]