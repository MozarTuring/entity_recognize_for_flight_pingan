# from infer import *
import time
import json
import flask
from flask import request
# from werkzeug.contrib.fixers import ProxyFix
import re
from flask import jsonify
from rule_based.flight_extraction import flight_extraction
from rule_based.flight_pattern import generate_pattern
from ocr_rectify.utils.get_position import *
from constants_jw import *
from rule_based.config_rule import *
from utils_mjw import post_func, extractFee_from_ocr_result, postprocess_dic
import requests
import multiprocessing as mp
from mrc.mrc import mrc_initialize, mrc_extract
from IE_evaluation_utils import entity_assign2catego, ATTACH_SYMBOL
import logging
import ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)  # if args.local_rank in [-1, 0] else logging.WARN

server = flask.Flask(__name__)


@server.route('/health', methods=['get'])
def health():
    return 'I am fine'


@server.route('/base64', methods=['post'])
def service_for_base64():
    start_time = time.time()
    data_get = json.loads(request.data.decode())
    logger.info('Received data: {}'.format(list(data_get.keys())))

    if isinstance(data_get['base64'], list):
        base64_ls = data_get['base64']
    else:
        base64_ls = [data_get['base64']]

    r_ls = [0 for _ in range(len(base64_ls))]

    def request_func(data_sent2ocr, ind):
        r = requests.post(OCR_URL, json.dumps(data_sent2ocr))
        # logger.info(r)
        r_ls[ind] = r

    data_sent2ocr = {
        'file_type': 'base64', 'token': OCR_TOKEN,
        'request_id': '1a645940-93a1-11e8-ba0b-06bd3c017f71', 'system': 'PA_TEST_CORE'}
    workers = []
    logger.info('Before ocr time consumed {} '.format(time.time() - start_time))
    start_time = time.time()
    logger.info('start ocr')
    for ind, ele in enumerate(base64_ls):
        data_sent2ocr['file'] = ele
        # logger.info(data_sent2ocr)
        request_func(data_sent2ocr, ind)
    #     workers.append(mp.Process(target=request_func, args=(data_sent2ocr, ind)))
    #     workers[-1].daemon = True
    #     workers[-1].start()
    # for worker in workers:
    #     worker.join()
    logger.info('Complete ocr in {}'.format(time.time() - start_time))
    sentences, data_rets, ocr_succeed_inds, r_dict_ls = extract_from_r_ls(r_ls)

    return get_rets(sentences, data_get['task_name'], start_time, data_rets, ocr_succeed_inds, ocr_result_ls=r_dict_ls)


@server.route('/sentence', methods=['post'])
def service_for_sentence():
    start_time = time.time()
    data_get = json.loads(request.data.decode())
    logger.info('Received data: {}'.format(list(data_get.keys())))

    if PRD:
        data_rets = [{"result": {}} for _ in range(len(data_get['sentences']))]
    else:
        data_rets = [{"result": {}, "zh_result": {}, "ocr_result": ""} for _ in range(len(data_get['sentences']))]

    return get_rets(data_get['sentences'], data_get['task_name'], start_time, data_rets)


@server.route('/ocr_ret', methods=['post'])
def service_for_ocr_ret():
    start_time = time.time()
    data_get = json.loads(request.data.decode())
    logger.info('Received data: {}'.format(list(data_get.keys())))

    sentences, data_rets, ocr_succeed_inds, r_dict_ls = extract_from_r_ls(data_get['r_ls'])

    return get_rets(sentences, data_get['task_name'], start_time, data_rets, ocr_succeed_inds, ocr_result_ls=r_dict_ls)


logger = logging.getLogger(__name__)
# ner_infer_discharge_records = NerInfer(**CONFIG_DIC['discharge_records'])
mrc_estimator, mrc_tokenizer = mrc_initialize()


def extract_from_r_ls(r_ls):
    if PRD:
        data_rets = [{"result": {}} for _ in range(len(r_ls))]
    else:
        data_rets = [{"zh_result": {}, "result": {}, "ocr_result": ""} for _ in range(len(r_ls))]
    ocr_succeed_inds = []
    r_dict_ls = []
    logger.info(r_ls)
    sentences = []
    for ind, r in enumerate(r_ls):
        if r:
            if isinstance(r, dict):
                r_dict = r
            else:
                r_dict = json.loads(r.text)
            # logger.info(r_dict)

            if r_dict['result']['message'] == 'success' and r_dict['error'] == '':
                # out_dic = {}
                # info_dic = {}
                info = r_dict['result']['info']
                if len(info) >= 3:
                    out = row_combine_new(info)
                    out = order_rectify(out)
                    out = split_section_simple(out, info)
                    out = generate_content(out)
                else:
                    out = ''
                    for dictt in info:
                        out = out + dictt['text'] + '\n'

                # out_dic['cur_pic'] = out
                # info_dic['cur_pic'] = info
                # sentences = [out]
                sentences.append(out)
                ocr_succeed_inds.append(ind)
                r_dict_ls.append(r_dict)
            else:
                logger.info("ocr结果异常: {}".format(r_dict))
                data_rets[ind] = {"code": "10006", "message": "ocr结果异常", "result": r_dict}
        else:
            logger.info("ocr接口返回值为空")
            data_rets[ind] = {"code": "10002", "message": "ocr接口返回值为空", "result": {}}

    return sentences, data_rets, ocr_succeed_inds, r_dict_ls


def get_rets(sentences, task_name, start_time, data_rets, succeed_inds=None, ocr_result_ls=None):
    new_sentences = []
    for i in sentences:
        # if "李小林" in i:
        #     import ipdb;ipdb.set_trace()
        tmp = re.sub("\n", 'mjwemptymjw', i)
        tmp = re.sub(r"\s{1,100}", "", tmp)
        # tmp = re.sub(r"\s{1,100}", "^", i) # use this one if there are many \s in train set and they are replaced by '^' at augmentation phase
        new_sentences.append(tmp)

    ner_infer, label_map_name, catego_short, entity_e2c, entity_c2e, mrc_questions = initialize(task_name)

    try:
        mrc_ret = mrc_extract(new_sentences, mrc_estimator, mrc_tokenizer, mrc_questions, ENTITY_C2E=entity_c2e)

        entity_list = []
        for i in range(len(new_sentences)):
            entity_lls = [[]]
            # self.catego_ls += ["age", "sex", "bed_number"]
            for k, v in PAT_DICT["出院记录"].items():
                res = v.findall(new_sentences[i])
                if res:
                    for ele in res:
                        if k in ["age", "sex", "bed_number"]:
                            ele = ele.replace("年龄", "").replace(":", "").replace("：", "").replace("岁", "").replace(
                                "性别", "").replace("床", "").replace("位", "").replace("号", "").replace("住", "")
                        entity_lls[0].append(k + ATTACH_SYMBOL + ele)

            # self.catego_ls += ["ad_diag", "dis_diag", "ad_main", "dis_situ", "dis_note"]

            for ques in mrc_questions:
                ele = mrc_ret["demo_" + str(i) + "_query_" + ques.replace("是什么？", "")]
                if ele:
                    entity_lls[0].append(
                        ENTITY_C2E_DICT['discharge_records.json'][ques.replace("是什么？", "")] + ATTACH_SYMBOL + ele)

            entity_list.append(entity_assign2catego(entity_lls, catego_short, tup=True))

        results = entity_list
        for ind, result in enumerate(results):
            temp_dic = get_dic(catego_short, result, new_sentences[ind], label_map_name)
            if "xingchengdan" in label_map_name and ocr_result_ls:
                extractFee_from_ocr_result(ocr_result_ls[ind], temp_dic)
            postprocess_dic(temp_dic)
            # bad_ocr = get_bad_ocr(catego_short, temp_dic, info)
            if succeed_inds:
                tmp_ind = succeed_inds[ind]
            else:
                tmp_ind = ind

            update_data_ret(temp_dic, catego_short, data_rets[tmp_ind], entity_e2c, new_sentences[ind].replace('mjwemptymjw', '\n'))
        logger.info('{}'.format(data_rets))
    except:
        if "code" not in data_rets[0]:
            data_rets = [{"code": "10008", "message": "字段抽取异常", "result": {}}]
        logger.info("{}".format(data_rets))

    # print(data_rets)
    logger.info('Time consumed: {}'.format(time.time() - start_time))
    # if len(data_rets) == 1:
    #     data_rets = json.loads(json.dumps(data_rets[0]).replace('^', ' '))
    # else:
    data_rets = json.loads(json.dumps(data_rets).replace('^', ' '))

    return jsonify(data_rets)


def initialize(task_name):
    label_map_name = task_name + '.json'
    catego_short = CATEGO_DICT[label_map_name]['short'] + CATEGO_DICT[label_map_name]['long']
    entity_e2c = ENTITY_E2C_DICT[label_map_name]
    entity_c2e = ENTITY_C2E_DICT[label_map_name]
    mrc_questions = [ele + '是什么？' for ele in entity_c2e.keys()]

    ret_infer = None
    # if task_name == 'xingchengdan':
    #     ret_infer = ner_infer_xingchengdan
    # if task_name == 'boardcards':
    #     ret_infer = ner_infer_boardcards
    # if task_name == 'xingchengdan_3code':
    #     ret_infer = ner_infer_xingchengdan_3code
    # if task_name == "discharge_records":
    #     ret_infer = ner_infer_discharge_records

    return ret_infer, label_map_name, catego_short, entity_e2c, entity_c2e, mrc_questions


def extract_info_for_board_card(source_data):
    flight_code_pattern = generate_pattern(file_path=code_path, key=code_key, pattern=code_pattern)
    flight_date_pattern = generate_pattern(file_path=date_path, key=date_key, pattern=date_pattern)
    flight_number, _ = flight_extraction(source_data, flight_code_pattern, flight_code_replace)
    flight_date, _ = flight_extraction(source_data, flight_date_pattern, flight_date_replace)

    return {'Flight_Number': flight_number, 'Flight_Date': flight_date}


def get_bad_ocr(catego_short, inp_dic, info):
    threshold = dict(zip(catego_short, [0 for _ in range(len(catego_short))]))
    ret_dic = dict(zip(catego_short, [[] for _ in range(len(catego_short))]))
    for catego in catego_short:
        if catego in ['Flight_Number', 'Flight_Date'] or not inp_dic[catego]:
            continue
        for pred in inp_dic[catego]:
            for ele in info:
                if pred.split('-')[-1] in ele['text']:
                    if ele['score'] < threshold[catego]:
                        ret_dic[catego].append(pred)
                        break


def get_dic(catego_short, result, inp_sentence, label_map_name):
    # for cur_pic, out in out_dic.items():
    ret_dic = dict(zip(catego_short, [[] for _ in range(len(catego_short))]))

    # pred_entity_rule = extract_info_for_board_card(inp_sentence.replace('\n', ''))
    # for catego in catego_short:
    #     if pred_entity_rule.get(catego, None) and catego == 'Flight_Date':
    #         ret_dic[catego] = ['0-'+pred_entity_rule[catego]]

    for catego in catego_short:
        if 'Fee' in catego:
            continue
        # if catego in ['Flight_Date']:
        #     if ret_dic[catego]:
        #         continue

        for value in result[catego][0]:
            value = value.replace('^', '')
            value = post_func(value.upper(), inp_sentence, 'interface', catego, label_map_name=label_map_name)

            if value:
                # value = value.replace('-', '$@#')
                if value not in ret_dic[catego]:
                    try:
                        ret_dic[catego].append(value)
                    except:
                        import ipdb; ipdb.set_trace()
    return ret_dic


def update_data_ret(inp_dic, catego_short, data_ret, entity_e2c, ocr_result, bad_ocr=None):
    for catego in catego_short:
        # 对于拼音类字段，如果它对应的非拼音字段有抽取结果并且抽取结果的ocr置信度高于设定阈值，那么不返回该拼音类字段
        # if 'PY' in catego and inp_dic[catego.split('_')[0]]:
        #     if bad_ocr:
        #         if bad_ocr[catego.split('_')[0]]==[]:
        #             continue
        #     continue

        if inp_dic[catego] or ('三字码' in catego):
            temp_ls = []
            for ele in inp_dic[catego]:
                temp_ls.append('{}'.format(ele))  # .split('-')[-1]
                if bad_ocr:
                    if ele in bad_ocr[catego]:
                        temp_str += ', ocr低置信度; '
            temp_str = ' '.join(temp_ls)
            temp_str = temp_str.replace('mjwemptymjw', '\n').replace('MJWEMPTYMJW', '\n')
            data_ret["result"].update({catego: temp_str})
            if not PRD:
                data_ret["zh_result"].update({entity_e2c[catego]: temp_str})  # .replace('$@#', '-')
                data_ret["ocr_result"] = ocr_result

        elif not inp_dic[catego]:
            data_ret["result"].update({catego: ""})
            if not PRD:
                data_ret["zh_result"].update({entity_e2c[catego]: ""})
                data_ret["ocr_result"] = ocr_result

    data_ret.update({"code": "200", "message": "字段抽取成功"})


# server.wsgi_app = ProxyFix(server.wsgi_app)

if __name__ == '__main__':
    server.config['JSON_AS_ASCII'] = False
    server.run(debug=False, port=53561, host='0.0.0.0')  # 注意部署到神兵的时候，端口号要和神兵环境设置的一致 55660， 不能大于65535
