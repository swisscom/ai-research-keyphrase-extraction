# Use a base image that comes with NumPy and SciPy pre-installed
FROM publysher/alpine-scipy:1.0.0-numpy1.14.0-python3.6-alpine3.7
# Because of the image, our versions differ from those in the requirements.txt:
#   numpy==1.14.0 (instead of 1.13.1)
#   scipy==1.0.0 (instead of 0.19.1)

# Install Java for Stanford Tagger
RUN apk --update add openjdk8-jre
# Set environment
ENV JAVA_HOME /opt/jdk
ENV PATH ${PATH}:${JAVA_HOME}/bin

# Download CoreNLP full Stanford Tagger for English
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip && \
    unzip stanford-corenlp-full-*.zip && \
    rm stanford-corenlp-full-*.zip && \
    mv stanford-corenlp-full-* stanford-corenlp

# Install sent2vec
RUN apk add --update git g++ make && \
    git clone https://github.com/epfml/sent2vec && \
    cd sent2vec && \
    git checkout f827d014a473aa22b2fef28d9e29211d50808d48 && \
    make && \
    apk del git make && \
    rm -rf /var/cache/apk/* && \
    pip install cython && \
    cd src && \
    python setup.py build_ext && \
    pip install .



# Install requirements
WORKDIR /app
ADD requirements.txt .
# Remove NumPy and SciPy from the requirements before installing the rest
RUN cd /app && \
    sed -i '/^numpy.*$/d' requirements.txt && \
    sed -i '/^scipy.*$/d' requirements.txt && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Set the paths in config.ini
ADD config.ini.template config.ini
RUN sed -i '6 c\host = localhost' config.ini && \
    sed -i '7 c\port = 9000' config.ini && \
    sed -i '10 c\model_path = /sent2vec/pretrained_model.bin' config.ini

# Add actual source code
ADD swisscom_ai swisscom_ai/
ADD launch.py .

ENTRYPOINT ["/bin/sh"]