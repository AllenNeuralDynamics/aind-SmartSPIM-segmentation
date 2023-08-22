FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n cell_seg python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cell_seg", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    bokeh==2.4.2 \
    dask[distributed]==2022.11.1 \
    dask_memusage \
    argschema==3.0.4 \
    natsort==8.2.0 \
    astropy==5.2.1 \
    photutils==1.6.0 \
    imlib==0.1.3 \
    zarr==2.13.6 \
    aind-data-schema==0.12.5 \
    aind-ng-link==1.0.6 \
    jupyter-server-proxy==3.2.2 \
    git+https://github.com/camilolaiton/cellfinder-core.git@block_seg

RUN conda install -c conda-forge awscli 
