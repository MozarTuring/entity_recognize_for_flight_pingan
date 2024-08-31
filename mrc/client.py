# import grpc
# import tensorflow as tf
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc
# import logging


tf_serving_url = "http://mjw-tf-serving-gpu-stg.icore-aas-rbt-gpu-stg.paic.com.cn/v1/models/aaaa:predict"

# logger = logging.getLogger(__name__)


# def get_grpc(inp):
#     # 设置grpc
#     options = [('grpc.max_send_message_length', 1000 * 1024 * 1024),
#                ('grpc.max_receive_message_length', 1000 * 1024 * 1024)]
#     channel = grpc.insecure_channel(tf_serving_url, options=options)
#     stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'aaaa'
#     request.model_spec.signature_name = 'serving_default'
#
#     for k, v in inp.items():
#         tmp_tensor = tf.contrib.util.make_tensor_proto(v)
#         request.inputs[k].CopyFrom(tmp_tensor)
#
#     # start = time.time()
#
#     # 法一，速度较慢
#     # result = stub.Predict(request, 10.0)  # 10 secs timeout
#
#     # 法二，速度较快
#     result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
#     logger.info('{}'.format(result_future))
#     result = result_future.result()
#     return result
