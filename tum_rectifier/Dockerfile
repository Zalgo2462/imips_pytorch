FROM ubuntu:16.04
ARG UID
ARG GID

RUN apt-get update && apt-get -y install libeigen3-dev libopencv-dev git cmake wget unzip
WORKDIR /root
RUN git clone https://github.com/tum-vision/mono_dataset_code.git
WORKDIR /root/mono_dataset_code
RUN cd thirdparty/ && tar -zxvf libzip-1.1.1.tar.gz && cd libzip-1.1.1/ && ./configure && make && make install && cp lib/zipconf.h /usr/local/include/zipconf.h
RUN cmake . && make && mv bin/* /usr/local/bin
WORKDIR /root
RUN mkdir data
WORKDIR /root/data

ENV UID ${UID}
ENV GID ${GID}

CMD if [ ! -f all_sequences.zip ]; then wget --progress=dot http://vision.in.tum.de/mono/all_sequences.zip; fi; unzip -n all_sequences.zip; for dir in *; do if [ -d $dir ]; then mkdir -p $dir/rect; cd $dir/rect; playDataset ../ dummy; cd ../../; fi; done; chown -R ${UID}:${GID} *
