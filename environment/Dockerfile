FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    antspyx==0.3.4 \
    argschema==3.0.4 \
    dask==2022.10.2 \
    s3fs==2022.11.0 \
    scikit-image==0.19.3 \
    tifffile==2022.10.10 \
    zarr==2.13.3

RUN conda install -c conda-forge awscli