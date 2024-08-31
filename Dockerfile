# FROM hub.yun.paic.com.cn/icore-bdas-ai/python36:base
FROM hub.yun.paic.com.cn/wiseapm/python:2.0.0.0-icore-bdas-ai-python36-cuda9.0

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' > /etc/timezone
RUN locale
#RUN docker version
# RUN localedef -f UTF-8 -i en_US en_US.utf8
# RUN cat /etc/sysconfig/i18n
# RUN echo 'LANG=en_US.UTF-8' > /etc/sysconfig/i18n
# RUN echo 'LC_ALL=en_US.UTF-8' >> /etc/sysconfig/i18n
# RUN cat /etc/sysconfig/i18n
# RUN source /etc/sysconfig/i18n
ENV LANG     en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL   en_US.UTF-8
RUN locale

# RUN ls /usr/share/i18n/charmaps/
# RUN export LC_ALL=en_US.UTF-8


COPY ./requirements.txt /tmp/

RUN python3 -m pip install --upgrade pip -i http://mirrors.yun.paic.com.cn:4048/pypi/web/simple  --trusted-host mirrors.yun.paic.com.cn
RUN python3 -m pip install -r /tmp/requirements.txt -i http://mirrors.yun.paic.com.cn:4048/pypi/web/simple  --trusted-host mirrors.yun.paic.com.cn ; \
    rm -rf /root/.cache/ ; \
    mkdir -pv /app

COPY ./ /app

WORKDIR /app

ENTRYPOINT python3 flight_ner_interface_base64.py