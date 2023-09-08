FROM python:3.6-slim-buster

ARG FUNCTION_DIR="/home/app/"
ARG LOCAL_FUNC_DIR
ARG ADDITIONAL_PACKAGE

RUN apt-get -qy update && apt-get -qy install gcc make libpng-dev libjpeg-dev ${ADDITIONAL_PACKAGE}

# Add non root user
RUN addgroup --system app && adduser app --system --ingroup app
RUN chown app /home/app

RUN mkdir -p /cuda_dumps
RUN chmod a+rw -R /cuda_dumps

COPY template/onnx_opt/ /usr/local/cuda
RUN mkdir -p /usr/local/cuda/bin
COPY template/cuobjdump  /usr/local/cuda/bin/cuobjdump
RUN ls -la /usr/local/cuda
RUN ls -la /usr/local/cuda/bin

# what a hack...
RUN mkdir -p /home/ubuntu/serverless-gpus/src/apps/covidct/refactored
COPY template/zoom_kernel.cubin /home/ubuntu/serverless-gpus/src/apps/covidct/refactored
RUN chmod a+r /home/ubuntu/serverless-gpus/src/apps/covidct/refactored/zoom_kernel.cubin

RUN mkdir -p plibs
COPY template/plibs/*.whl  plibs/
RUN python3 -m pip install plibs/*.whl
RUN rm -rf plibs

# always run stuff below this random
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Copy handler function
COPY ${LOCAL_FUNC_DIR} ${FUNCTION_DIR}

COPY template/ava.conf /home/app/
RUN cat /home/app/ava.conf

USER app

WORKDIR ${FUNCTION_DIR}

ENV PATH=$PATH:/home/app/.local/bin

# Install the function's dependencies
RUN python3 -m pip install -r requirements.txt --target ${FUNCTION_DIR}

# Install Lambda Runtime Interface Client for Python
RUN python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
COPY template/entry.sh /

USER root
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh

USER app
ENTRYPOINT [ "/entry.sh" ]
CMD [ "app.handler" ]
