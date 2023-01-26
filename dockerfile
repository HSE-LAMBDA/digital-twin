FROM python:3.9-bullseye
# Set user jovyan
RUN useradd -ms /bin/bash jovyan
USER jovyan
# Set working directory
WORKDIR /home/jovyan
