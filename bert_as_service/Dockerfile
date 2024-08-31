#FROM hub.yun.paic.com.cn/wiseapm/python:2.0.0.0-icore-bdas-ai-python36-cuda9.0
FROM hub.yun.paic.com.cn/tensorflow/tensorflow:1.12.0-python36-cuda9.0-cudnn7-devel-ubuntu16.04

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' > /etc/timezone

#ENV LANG     en_US.UTF-8
#ENV LANGUAGE en_US.UTF-8
#ENV LC_ALL   en_US.UTF-8

#ENV NVIDIA_VISIBLE_DEVICES 2

COPY ./requirements.txt /tmp/

RUN python3 -m pip install --upgrade pip -i http://mirrors.yun.paic.com.cn:4048/pypi/web/simple  --trusted-host mirrors.yun.paic.com.cn
RUN python3 -m pip install -r /tmp/requirements.txt -i http://mirrors.yun.paic.com.cn:4048/pypi/web/simple  --trusted-host mirrors.yun.paic.com.cn ; \
    rm -rf /root/.cache/

ENTRYPOINT bert-serving-start -http_port 8125 -device_map 1 -num_worker 1 -model_dir /root/maojingwei579/chinese_L-12_H-768_A-12

#RUN mkdir -p /app
#COPY ./entrypoint.sh /app
#
#ENTRYPOINT ["/app/entrypoint.sh"]