FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y gosu sudo locales

RUN find /root -type f -print0 | xargs -0 chmod 666 \
      && find /root -type d -print0 | xargs -0 chmod 777
RUN echo "export PS1='\\W$ '" >> /root/.bashrc
ENV HOME=/root
# Yes, allow anyone to run as root with gosu
RUN chmod gu+s `which gosu`

#install ava deps
COPY ava/install_dependencies.sh /tmp
RUN /tmp/install_dependencies.sh

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

VOLUME /root/.ccache
VOLUME /source
WORKDIR /source

ARG USER_ID
ARG GROUP_ID

#RUN addgroup --gid $GROUP_ID user
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get install -y tmux zsh tig
#copied below from ava tools/install_dependencies.sh
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt update -y
RUN apt purge --auto-remove cmake -y
RUN apt install -y cmake cmake-curses-gui
RUN apt install -y llvm-9
ENV PATH="${PATH}:/usr/lib/llvm-9/bin"
RUN sudo mkdir /cuda_dumps

#install go
RUN wget https://golang.org/dl/go1.16.5.linux-amd64.tar.gz \
    && tar -C /usr/local -xf go1.16.5.linux-amd64.tar.gz
ENV PATH="${PATH}:/usr/local/go/bin:/root/go/bin"

#install task
RUN wget https://github.com/go-task/task/releases/download/v3.4.3/task_linux_amd64.deb \
&& dpkg -i task_linux_amd64.deb

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t agnoster

RUN echo set-option -g default-shell /bin/zsh >> /root/.tmux.conf

RUN mkdir -p /root/.oh-my-zsh/comp
RUN wget https://raw.githubusercontent.com/go-task/task/master/completion/zsh/_task && sudo cp _task /root/.oh-my-zsh/comp/_task
RUN echo 'fpath=(/root/.oh-my-zsh/comp $fpath)' | sudo tee -a /root/.zshrc
RUN echo 'alias sudo="gosu root"' | sudo tee -a /root/.zshrc
RUN echo 'autoload -U compinit && compinit' | sudo tee -a /root/.zshrc

RUN mkdir -p /root/.ssh
RUN echo "StrictHostKeyChecking no\n" >> /root/.ssh/config

#install grpc and protoc
RUN go get github.com/golang/protobuf/proto
RUN go get github.com/golang/protobuf/protoc-gen-go
RUN go get google.golang.org/grpc
RUN go get google.golang.org/grpc/cmd/protoc-gen-go-grpc

run sudo apt install unzip
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip \
    && mkdir protoc && unzip protoc-3.17.3-linux-x86_64.zip -d protoc \
    && chmod -R a+rx protoc \
    && sudo mv protoc/bin/* /usr/bin/ && sudo mv protoc/include/* /usr/include/ \
    && rm -rf protoc

RUN git clone --recurse-submodules -b v1.38.0 https://github.com/grpc/grpc \
    && cd grpc && mkdir -p cmake/build && cd cmake/build && cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../.. \
    && make -j && sudo make install \
    && cd - && mkdir -p third_party/abseil-cpp/cmake/build && cd third_party/abseil-cpp/cmake/build \
    && cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ../.. && make -j && sudo make install

#fix broken permissions
RUN sudo chown -R $USER_ID:$GROUP_ID /root/.cache
RUN sudo chown -R $USER_ID:$GROUP_ID /root/go

#for covidct workload
RUN apt install -y swig
#RUN python3 -m pip install scikit-image==0.17.2 keras==2.2.5 h5py<3.0.0 cucim==21.6.0 cupy-cuda101==9.2.0 pydicom==2.1.2 python-gdcm
RUN python3 -m pip install --upgrade pip 
RUN python3 -m pip install --upgrade setuptools
RUN python3 -m pip install --ignore-installed PyYAML
RUN python3 -m pip install numba scikit-build pybind11 opencv-python-headless requests cython scikit-image scikit-learn transformers pycocotools

RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.10.0/bazelisk-linux-amd64 \
    && chmod +x bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/local/bin \
    && cd /usr/local/bin && ln -s bazelisk-linux-amd64 bazel

RUN echo set-option -g history-limit 5000 >> /root/.tmux.conf
