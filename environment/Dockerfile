# hash:sha256:e70a5cf1e49dd28853e53079c59d19d1decbfe6acf15b7102902e44a2b66a349
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n cell_seg python=3.10

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cell_seg", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libnvrtc11.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    cupy-cuda11x==13.4.0 \
    scipy==1.15.2 \
    bokeh==2.4.2 \
    dask[distributed]==2024.4.1 \
    dask_memusage \
    argschema==3.0.4 \
    natsort==8.2.0 \
    zarr==2.18.0 \
    aind-data-schema==1.0.0 \
    awscli

