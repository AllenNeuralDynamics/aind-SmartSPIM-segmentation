# hash:sha256:efe04cd2567bb9241fc6b72a3fb36d96271710494fb18df0bd31e189420457d0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libnvrtc11.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git@py38-compatible \
    numpy==1.24.4 \
    scikit-image==1.3.2 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    cupy-cuda11x==13.4.0 \
    neuroglancer==2.40.1 \
    aind-data-schema==1.2.0 \
    aind-ng-link==1.0.17