sudo apt-get install -y git python-pip python-dev
sudo apt-get install -y python-dev
sudo apt-get install -y autoconf automake libtool curl make g++ unzip
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

# install torch
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc

# install pytorch
mkdir Projects
cd Projects
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
source ~/torch/install/bin/torch-activate
./build.sh

sudo apt-get install -y redis-server rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
sudo service rabbitmq-server restart 
sudo service redis-server restart

# install nginx
sudo apt-get update
sudo apt-get install -y python-software-properties
sudo add-apt-repository ppa:nginx/development
sudo apt-get install -y nginx

luarocks install loadcaffe

# if gpu
luarocks install cudnn
luarocks install cunn

# cuda installation
# link to download cuda https://developer.nvidia.com/cuda-downloads
cd ..
mkdir cuda_installation
cd cuda_installation
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q -y
sudo apt-get -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install linux-generic
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo apt-get update -q -y
sudo apt-get install cuda -q -y

# install cudnn 
export CUDA_HOME=/usr/local/cuda-7.5 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-370

#install cudnn from nvidia site and then run the following command
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH

#install dependencies
cd ..
git clone https://github.com/Cloud-CV/Grad-CAM.git
cd Grad-CAM

git submodule init && git submodule update
sh models/download_models.sh
pip install -r requirements.txt

python manage.py collectstatic
