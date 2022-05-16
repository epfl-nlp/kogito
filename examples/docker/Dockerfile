FROM python

ARG GITHUB_PERSONAL_TOKEN

ENV HOME=/root
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV KOGITO_DIR=${HOME}/kogito
ENV POETRY=${HOME}/.local/bin/poetry

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends vim wget unzip

WORKDIR ${HOME}

# Cluster setup
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O anaconda.sh
RUN bash anaconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN git config --global user.name "Mete Ismayil"
RUN git config --global user.email "mismayilza@gmail.com"
RUN echo "export LANG=en_US.UTF-8" >> ~/.bashrc

# Clone kogito
RUN git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/epfl-nlp/kogito.git

WORKDIR ${KOGITO_DIR}

# Setup kogito
RUN ${CONDA} create --name kogito -y python=3.8
RUN ${CONDA} run -n kogito curl -sSL https://install.python-poetry.org | python3 -
RUN ${CONDA} run -n kogito ${POETRY} install
RUN ${CONDA} install -n kogito pytorch cudatoolkit=11.3 -c pytorch

# Install training data
WORKDIR ${KOGITO_DIR}/examples/data
RUN wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
RUN unzip atomic2020_data-feb2021.zip

WORKDIR ${KOGITO_DIR}/examples

COPY ./entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]