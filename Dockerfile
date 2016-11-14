FROM andrewosh/binder-base:latest

# based on examples from binder and jupyter showcase
MAINTAINER Malte Vogl <malte.vogl@hu-berlin.de>

USER main

ENV DASHBOARDS_VERSION 0.6.1

ENV CMS_VERSION 0.6.1

# get to the latest jupyter release and necessary libraries
RUN conda install -y jupyter

# install incubator extensions
RUN pip install jupyter_dashboards==$DASHBOARDS_VERSION \
    jupyter_cms==$CMS_VERSION

RUN jupyter dashboards install --user --symlink && \
    jupyter cms install --user --symlink && \
    jupyter dashboards activate && \
    jupyter cms activate

# install kernel-side incubator extensions for python3 environment too
RUN bash -c "source activate python3 && pip install \
    jupyter_cms==$CMS_VERSION"

# Install requirements for Python
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install requirements for Python 3
RUN conda install -c conda-forge jupyter_contrib_nbextensions && \
    bash -c "jupyter nbextensions_configurator enable --user" && \
    bash -c "source activate python3 && \
        pip install -r requirements.txt"

RUN conda install -y -c conda-forge ipywidgets

RUN pip install -U textblob && python -m textblob.download_corpora lite
