FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean --all -y

# Activate conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

