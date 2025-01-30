FROM tensorflow/tensorflow:latest-gpu

USER root

COPY requirements.txt .
RUN python3 -m pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install graphviz -y

WORKDIR /root/_app
CMD ["tail", "-f", "/dev/null"]
