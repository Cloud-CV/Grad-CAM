# Start with CUDA Torch dependencies
FROM kaixhin/cuda-torch-deps:latest

MAINTAINER Deshraj <deshrajdry@gmail.com>

# Run Torch7 installation scripts
RUN cd /root/torch && \
  ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

RUN apt-get update
RUN apt-get install -y python-dev libhdf5-serial-dev  libprotobuf-dev protobuf-compiler

# Install PyTorch
RUN pip install numpy==1.11.1 pytest
RUN git clone https://github.com/hughperkins/pytorch.git && cd pytorch && ./build.sh

# Clone the repository
RUN git clone https://github.com/DESHRAJ/grad-cam.git

# Update git submodules
RUN cd grad-cam && git submodule init && git submodule update

# Change relative path in lua for neuraltalk2
RUN cd grad-cam && sed -i -e "s/local utils = require 'misc.utils'/local utils = require 'neuraltalk2.misc.utils'/g" neuraltalk2/misc/LanguageModel.lua
RUN cd grad-cam && sed -i -e "s/local net_utils = require 'misc.net_utils'/local net_utils = require 'neuraltalk2.misc.net_utils'/g" neuraltalk2/misc/LanguageModel.lua
RUN cd grad-cam && sed -i -e "s/local LSTM = require 'misc.LSTM'/local LSTM = require 'neuraltalk2.misc.LSTM'/g" neuraltalk2/misc/LanguageModel.lua

# Install python dependencies
RUN cd grad-cam && pip install -r requirements.txt
RUN python -m nltk.downloader all

# Install lua dependencies
RUN luarocks install loadcaffe
RUN luarocks install nn
RUN luarocks install cunn
RUN luarocks install cudnn

RUN apt-get install -y unzip wget

# Downlaod the models
RUN cd grad-cam && bash models/download_models.sh && pwd

WORKDIR /grad-cam

EXPOSE 80
EXPOSE 8000

CMD  ["/bin/bash", "/grad-cam/Docker/deploy.sh"]
