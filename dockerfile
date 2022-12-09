FROM python:3.11.1-bullseye
# Set user jovyan
RUN useradd -ms /bin/bash jovyan
USER jovyan
# Set working directory
WORKDIR /home/jovyan
