FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04
ENV TZ=Asia/Hong_Kong
ARG DEBIAN_FRONTEND=noninteractive

RUN chsh -s /bin/bash

SHELL ["/bin/bash", "-c"]

WORKDIR /root/

ARG CACHE_DATE=2023-03-31
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    wget \
    bzip2 \
    apt-utils \
    git \
    g++ \
    libeigen3-dev \
    wget \
    cmake \
    less \
    unzip
RUN apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./environment.yml ./environment.yml

RUN conda update conda \
	&& conda env create --name diffdock -f ./environment.yml

RUN echo "conda activate diffdock" >> ~/.bashrc
ENV PATH /opt/conda/envs/diffdock/bin:$PATH
ENV CONDA_DEFAULT_ENV $diffdock

#install torch specific packages
RUN pip install --upgrade pip 
COPY ./requirements_docker_GPU_oddt.txt ./

RUN pip install --no-cache-dir torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir -r ./requirements_docker_GPU_oddt.txt

RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-geometric torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

COPY . .
RUN git submodule init
RUN git submodule update
RUN pip install -e ./esm/.

RUN python datasets/esm_embedding_preparation.py --protein_path test/test.pdb --out_file data/prepared_for_esm.fasta && \
    HOME=/app/esm/model_weights python esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm.fasta data/esm2_output --repr_layers 33 --include per_tok && \
    python -m inference --protein_path test/test.pdb --ligand test/test.sdf --out_dir /outputs --inference_steps 20 --samples_per_complex 40 --batch_size 10 --actual_steps 18 --no_final_step_noise

CMD ["bash"]
