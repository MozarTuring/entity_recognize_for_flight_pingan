#FROM hub.yun.paic.com.cn/tensorflow/serving:1.12.0-gpu-tensorrt4.1-nccl2.2-cudnn7.2-cuda9.0-ubuntu16.04
FROM hub.yun.paic.com.cn/tensorflow/serving:1.13.0-gpu-tensorrt5-cudnn7-cuda10-ubuntu16.04
#FROM hub.yun.paic.com.cn/tensorflow/serving:2.1.0-ubuntu16.04

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone

#COPY ./tf_serving/models.config /models/

ENV NVIDIA_VISIBLE_DEVICES 3

#ENTRYPOINT /usr/bin/tf_serving_entrypoint.sh --model_config_file=/models/models.config --per_process_gpu_memory_fraction=25
ENTRYPOINT tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name="aaaa" --model_base_path="/root/maojingwei579/cmrc2018-master/baseline/discharge_records_0730/tf1_13_bcls/pb_models"