# import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
import ipdb
# import os
#
#
# def freeze_graph(model_folder, output_graph):
#     '''
#     :param input_checkpoint:
#     :param output_graph: PB模型保存路径
#     :return:
#     '''
#     checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
#     input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
#
#     # ipdb.set_trace()
#     # reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
#     # var_to_shape_map = reader.get_variable_to_shape_map()
#     # for key in var_to_shape_map:
#     #     print("tensor_name: ", key)
#
#     # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
#     output_node_names = "add,add_1,BiasAdd"
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     graph = tf.get_default_graph()  # 获得默认的图
#     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
#
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
#         output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
#             sess=sess,
#             input_graph_def=input_graph_def,  # 等于:sess.graph_def
#             output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
#
#         with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
#             f.write(output_graph_def.SerializeToString())  # 序列化输出
#         print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
#
#
# model_dir = os.environ.get("OUTPUT_DIR", "/root/maojingwei579/cmrc2018-master/baseline/discharge_records/short_long")
# output_path = os.path.join(model_dir, 'saved_model.pb')
# freeze_graph(model_dir, output_path)


#### 裁剪
# def save(self, bert_save, load_model_path, output_dir):
# 	  #input_idsList_, input_masksList_, segment_idsList_, labels_ = self.data_embedding(test_texts, test_labels)
# 	  g = tf.Graph()
# 	  with g.as_default():
# 		loss, predict, true_, acc = bert_save.create_model()
# 	  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g) as sess:
# 		# sess = tf.InteractiveSession(graph=g)
# 		# sess.run(tf.global_variables_initializer())
# 		# sess.run(tf.global_variables_initializer())
# 		saver = tf.train.Saver(max_to_keep=1)
# 		saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
# 		# a = [n.name for n in tf.get_default_graph().as_graph_def().node]
# 		# file = open('./a.txt', 'w+')
# 		# for i in range(len(a)):
# 		#     s = a[i] + '\n'
# 		#     file.write(s)
# 		# file.close()
# 		variables = get_variables_to_restore()
# 		other_vars = [variable for variable in variables if not re.search("Adam", variable.name)]
# 		var_saver = tf.train.Saver(other_vars)
# 		# valid_loss, valid_acc = self.predict_accuracy(sess, loss, acc, input_idsList_, input_masksList_,
# 		#							     segment_idsList_, labels_)
# 		# print('valid_acc:{0}, valid_loss:{1}'.format(valid_acc, valid_loss))
# 		var_saver.save(sess, output_dir + 'model' + str(int(time.time())))
# 		saver.save(sess, output_dir + 'model' + str(int(time.time())))
# 	  print('end...')



###### 转 PB
# import os
# import tensorflow as tf
#
#
# def ckpt2pb(dir_ckpt_path,dir_pb_path):
#     checkpoint_file = tf.train.latest_checkpoint(dir_ckpt_path)
#     graph = tf.Graph()
#
#     with graph.as_default():
# 	  session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# 	  sess = tf.Session(config=session_conf)
#
# 	  with sess.as_default():
# 		# 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
# 		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
# 		sess.run(tf.global_variables_initializer())
# 		saver.restore(sess, checkpoint_file)
#
# 		input_ids = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("input_ids").outputs[0])
# 		input_mask = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("input_mask").outputs[0])
# 		segment_ids = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("segment_ids").outputs[0])
# 		y_pred_cls = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("predictions").outputs[0])
# 		y_pred_proba = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("Softmax").outputs[0])
#
# 		# ckpt转成pd格式
# 		builder = tf.saved_model.builder.SavedModelBuilder(dir_pb_path)
#
# 		inputs = {
# 		    # 注意，这里是你预测模型的时候需要传的参数，调用模型的时候，传X参必须和这里一致
# 		    # 这里的input_x就是模型里面定义的输入placeholder
# 		    "input_ids": input_ids,
# 		    "input_mask": input_mask,
# 		    "segment_ids": segment_ids
# 		}
# 		outputs = {
# 		    "y_pred_proba": y_pred_proba,
# 		    "y_pred_cls": y_pred_cls
# 		}
# 		prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
# 		    inputs=inputs,
# 		    outputs=outputs,
# 		    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
# 		legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
# 		builder.add_meta_graph_and_variables(
# 		    sess,
# 		    [tf.saved_model.tag_constants.SERVING],
# 		    signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
# 		    legacy_init_op=legacy_init_op
# 		)
# 		builder.save()
# 		print('end...')




# -*- coding:utf-8 -*-
# @Time : 2020-3-16 16:53
# @Author : XIONGHAIQUAN366
import os, re
import tensorflow as tf
from tensorflow.contrib.slim import get_variables_to_restore
import ipdb


gpu_options = tf.GPUOptions(allow_growth=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def ckpt_to_pb(ckpt_dir, pt_dir=None, to_light=False):
    checkpoint_file = tf.train.latest_checkpoint(ckpt_dir)
    print(checkpoint_file)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_file)


            if to_light:
                print("start making model light...")
                variables = get_variables_to_restore()
                other_vars = [variable for variable in variables if not re.search("adam", variable.name)]
                var_saver = tf.train.Saver(other_vars)
                var_saver.save(sess, checkpoint_file.replace("model", "light_model"))
                return None

            print("start converting ckpt format to pb format...")
            input_ids = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("IteratorGetNext").outputs[0])
            input_mask = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("IteratorGetNext").outputs[1])
            segment_ids = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("IteratorGetNext").outputs[2])
            input_span_mask = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("IteratorGetNext").outputs[3])
            start_logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("add").outputs[0])
            end_logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("add_1").outputs[0])
            cls_logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("BiasAdd").outputs[0])

            # keep_rate = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("keep_rate").outputs[0])
            # probability = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("classifier/probability").outputs[0])

            # ckpt转成pb格式
            builder = tf.saved_model.builder.SavedModelBuilder(pt_dir)

            inputs = {
                # 注意，这里是你预测模型的时候需要传的参数，调用模型的时候，传参必须和这里一致
                # 这里的input_x就是模型里面定义的输入placeholder
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "input_span_mask": input_span_mask,
            }
            outputs = {
                "start_logits": start_logits,
                "end_logits": end_logits,
                "cls_logits": cls_logits
            }

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
                legacy_init_op=legacy_init_op,
            )
            builder.save()
            print("conversion finished !")

base_dir = "/root/maojingwei579/cmrc2018-master/baseline/discharge_records/short_long"

# saver = tf.train.Saver(max_to_keep=1)
# sess = tf.Session()
# saver.restore(sess, tf.train.latest_checkpoint(base_dir))
# variables = get_variables_to_restore()
# other_vars = [variable for variable in variables if not re.search("Adam", variable.name)]
# var_saver = tf.train.Saver(other_vars)
# var_saver.save(sess, os.path.join(base_dir, 'model'+ str(int(time.time()))))
# exit()
# ckpt_to_pb(base_dir, to_light=True)
ckpt_to_pb(base_dir, os.path.join(base_dir, "pb_models", "1"))