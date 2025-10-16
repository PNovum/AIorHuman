
FROM ghcr.io/mlflow/mlflow:latest
RUN pip install --upgrade pip && \
    pip install psycopg2-binary mlflow[auth] boto3
RUN mkdir -p /mlflow_config
COPY basic_auth.ini /mlflow_config/basic_auth.ini
WORKDIR /mlflow_config
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow_user:mlflow_pass@postgres:5432/mlflow_db", \
     "--default-artifact-root", "s3://mlflow", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--app-name", "basic-auth", \
     "--workers", "1"]