FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive


###############################################################################
# Dockerfile for https://github.com/ppwwyyxx/speaker-recognition
# -----------------------------------------------------------------------------
# Docker provides a way to run applications securely isolated in a container, 
# packaged with all its dependencies and libraries.
#
# This Dockerfile produces a docker image, from which containers can be created
# * An image is a lightweight, stand-alone, executable package that includes 
#   everything needed to run a piece of software, including the code, a runtime,
#   libraries, environment variables, and config files.
# * A container is a runtime instance of an image – what the image becomes in
#   memory when actually executed. It runs completely isolated from the host 
#   environment by default, only accessing host files and ports if configured 
#   to do so.
#
# Containers run apps natively on the host machine’s kernel. 
# They have better performance than virtual machines that only get virtual
# access to host resources through a hypervisor. 
# Images or containers can easily be exchanged and many users publish images in
# the docker hub (https://hub.docker.com/).  Docker further enables upscaling
# of solutions from single workstation to server farms through docker swarms.
#
#      Read more here: https://docs.docker.com/
# Install docker here: https://docs.docker.com/engine/installation/linux/
#
# Quick start commands (as root)
# -----------------------------------------------------------------------------
# Pull an image from the docker hub
# > docker pull <image name>
# 
# Build this Dockerfile (place it in an empty folder and cd to it): 
# > docker build -f Dockerfile -t speaker-recognition .
#
# Instantiate a container from an image
# > docker run -ti speaker-recognition
# To give container access to host files during development:
# > docker run --name speaker-recognitionInstance -ti -v /:/host speaker-recognition
#
# Run a stopped container
# > docker start -ai speaker-recognitionInstance
# 
# List information
# > docker images                 All docker images
# > docker ps -a                  All docker containers (running or not: -a)
#
###############################################################################


# Prepare package management
###############################################################################
RUN apt-get update && \
    apt-get install -y nano sudo tzdata apt-utils && \
    apt-get -y dist-upgrade


# Set timezone
# https://bugs.launchpad.net/ubuntu/+source/tzdata/+bug/1554806
###############################################################################
RUN rm /etc/localtime && echo "Australia/Sydney" > /etc/timezone && dpkg-reconfigure -f noninteractive tzdata


# Create the GUI User
###############################################################################
# Then you can run a docker container with access to the GUI on your desktop:
# > docker run -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -u guiuser <image>
# -----------------------------------------------------------------------------
ENV USERNAME guiuser
RUN useradd -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Replace 1000 with your user/group id
    usermod  --uid 1000 $USERNAME && \
    groupmod --gid 1000 $USERNAME


# Python 2
###############################################################################
RUN apt-get update && apt-get install -y python python-pip && \
    pip2 list --outdated | cut -d' ' -f1 | xargs -n 1 pip2 install --upgrade


# Base Dependencies
###############################################################################
RUN apt-get install -y portaudio19-dev libopenblas-base libopenblas-dev pkg-config git-core cmake python-dev liblapack-dev libatlas-base-dev libblitz0-dev libboost-all-dev libhdf5-serial-dev libqt4-dev libsvm-dev libvlfeat-dev  python-nose python-setuptools python-imaging build-essential libmatio-dev python-sphinx python-matplotlib python-scipy


# Spear
# https://gitlab.idiap.ch/bob/bob/wikis/Dependencies
# Takes a very long time to install python packages because compilation is happening in the background
###############################################################################
RUN pip2 install scipy scikit-learn scikits.talkbox numpy pyside pyssp PyAudio argparse h5py
RUN pip2 install bob.extension
RUN pip2 install bob.blitz
RUN pip2 install bob.core
RUN pip2 install bob.io.base
RUN pip2 install bob.bio.spear
RUN pip2 install bob.sp


# Realtime Speaker Recognition
# https://github.com/ppwwyyxx/speaker-recognition
###############################################################################
RUN cd ~/ && \
    git clone https://github.com/ppwwyyxx/speaker-recognition.git && \
    cd ~/speaker-recognition && \
    make -C src/gmm


# Clean up
###############################################################################
RUN apt-get clean &&apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
