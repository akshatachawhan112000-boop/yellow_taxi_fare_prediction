FROM python:3.10-slim
WORKDIR /app
COPY ./app
RUN pip install pyspark pandas pyarrow matplotlib 
CMD ["python","scripts.py"]
