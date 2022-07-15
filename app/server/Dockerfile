FROM python:3.8

ENV HOME=/root
ENV APP_DIR=${HOME}/kogito-server
ENV FLASK_APP=server

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends vim wget unzip

RUN mkdir ${APP_DIR}
WORKDIR ${APP_DIR}

# Copy server files
COPY . .

# Setup app dependencies
RUN pip3 install -r requirements.txt
RUN python3 -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')"
RUN python3 -m spacy download en_core_web_sm

EXPOSE 5000

ENTRYPOINT ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]