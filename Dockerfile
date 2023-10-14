FROM python:3

RUN apt-get update -y && apt-get upgrade -y && \
  apt-get install -y libglu1-mesa-dev
  

WORKDIR /post_mission_analysis

COPY main.py .
COPY object_discovery.py .
COPY datasets.py .
COPY networks.py .
COPY requirements.txt .
COPY dino_deitsmall16_pretrain.pth .
COPY datasets.py .
COPY dino ./dino
COPY outputs ./outputs
#COPY resized_output ./resized_output

RUN pip install -r requirements.txt

#CMD ["python", "main_tokencut_dataset_jervis.py" --path resized_output]
#ENTRYPOINT ["python","main_tokencut_dataset_jervis.py"]
