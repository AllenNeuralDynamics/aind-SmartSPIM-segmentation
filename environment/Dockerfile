FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    dask \
    argschema \
    natsort \
    astropy \
    photutils \
    imlib \
    zarr \
    git+https://github.com/camilolaiton/cellfinder-core.git@feature/lazy_reader

RUN conda install -c conda-forge awscli