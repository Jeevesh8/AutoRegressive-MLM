FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

#
# Install Miniconda in /opt/conda
#

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN pip install --upgrade pip
RUN pip install --upgrade jax jaxlib==0.1.60+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN mkdir /content
RUN mkdir /content/drive/
RUN mkdir /content/drive/MyDrive
RUN mkdir /content/drive/MyDrive/2SCL/
RUN mkdir /content/drive/MyDrive/2SCL/Argumentation/
RUN mkdir /content/drive/MyDrive/2SCL/Argumentation/first_batch_data/

RUN git clone https://github.com/Jeevesh8/AutoRegressive-MLM/
RUN cd AutoRegressive-MLM
RUN git checkout PreTrain-12Layer
RUN pip install -r requirements.txt
