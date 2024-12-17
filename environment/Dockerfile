FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    numpy==1.26.4 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    scipy==1.10.1 \
    cupy-cuda11x==13.3.0 \
    neuroglancer==2.40.1 \
    aind-data-schema==1.2.0
