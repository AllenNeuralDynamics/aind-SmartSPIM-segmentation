FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    bokeh==2.4.2 \
    dask[distributed]==2022.11.1 \
    argschema==3.0.4 \
    natsort==8.2.0 \
    astropy==5.2.1 \
    photutils==1.6.0 \
    imlib==0.1.3 \
    zarr==2.13.6 \
    aind-data-schema==0.12.5 \
    git+https://github.com/camilolaiton/cellfinder-core.git@feature/bkg_sub

RUN conda install -c conda-forge awscli
