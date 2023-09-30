FROM ubuntu:20.04 as base
RUN apt-get update
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install wget vim python3-pip python-dev build-essential sudo -y
WORKDIR /fs3e
COPY . ./
RUN pip3 install -r requirements.txt
