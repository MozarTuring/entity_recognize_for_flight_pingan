from constants_jw import PATH_DICT, ENTITY_C2E_DICT, CATEGO_DICT, ENTITY_E2C_DICT
import os, re
import chardet
import ipdb
# from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from utils_mjw import post_func, postprocess_ls, NEED_POSTPROCESS
import csv
import pandas as pd
import json
import copy
from tmp_constants import aa


# Variables inside the follwing pre-defined functions are all local 
def get_num(true_entity_lls, pred_entity_lls, to_set=True):
    pred_num, gold_num, correct_num = [0 for _ in range(3)]
    assert len(true_entity_lls) == len(pred_entity_lls)
    for ind, (true_entity_ls, pred_entity_ls) in enumerate(zip(true_entity_lls, pred_entity_lls)):
        pred_entity_ls = list(map(lambda x: x.upper(), pred_entity_ls))
        true_entity_ls = list(map(lambda x: x.upper(), true_entity_ls))
        if to_set:
            # 这里的 to set 可以合并同一个样本中的同一个catego的相同entity
            pred_num += len(set(pred_entity_ls))
            gold_num += len(set(true_entity_ls))
        else:
            pred_num += len(pred_entity_ls)
            gold_num += len(true_entity_ls)
        correct_num += len(set(pred_entity_ls)&set(true_entity_ls)) 
    print(correct_num, pred_num, gold_num) 
    

# Variables inside the follwing pre-defined functions are all local or outer variables defined inside this box

ATTACH_SYMBOL = '+=+' 

def extract_entity(filepath=None, text_lls=None, label_lls=None):
    if filepath:
        with open(filepath, 'r', encoding='utf8') as rf:
            lines = rf.readlines()
        label_lls, label_ls, text_lls, text_ls = [[] for _ in range(4)]
        for line in lines:
            if line == '\n':
                assert len(label_ls) == len(text_ls)
                label_lls.append(label_ls)
                text_lls.append(text_ls)
                label_ls, text_ls = [], []
                continue

            label_ls.append(line.strip('\n').split(' ')[-1])
            text_ls.append(line.strip('\n').split(' ')[0])

    entity_lls = []
    for ind, label_ls in enumerate(label_lls):
        entity = ''
        entity_ls = []
        flag = False
        # label_str = ' '.join(label_ls)
        # pat = re.compile(r'B\-Hospital_Name[ I\-Hospital_Name]+ B\-Hospital_Name')
        # re.sub(pat, 'B\-Hospital_Name[ I\-Hospital_Name]+ B\-Hospital_Name', label_str)
        # label_ls = label_str.split(' ')
        for subind, label in enumerate(label_ls):
            if subind >= 1 and label == 'B-Hospital_Name':
                if label_ls[subind-1] == 'I-Hospital_Name':
                    label = 'I-Hospital_Name'
            cur_token = text_lls[ind][subind]
    #         if cur_token.lower() == '<pad>' or cur_token.lower() == '[pad]':
    #             break
            if 'B-' in label:
                flag = True
                if entity == '':
                    entity = label.replace('B-', '')+ATTACH_SYMBOL+cur_token
                else:
                    entity_ls.append(entity)
                    entity = label.replace('B-', '')+ATTACH_SYMBOL+cur_token
            elif 'I-' in label: # 实体中间编号，根据实际修改
                if flag:
                    entity = entity + cur_token
            else:
                flag = False
        if entity != '':
            entity_ls.append(entity)
        entity_lls.append(entity_ls)
    # if return_label_lls:
    #     return entity_lls, label_lls
    # else:
    #     return entity_lls
    return (entity_lls, label_lls, text_lls)

catego_ls = None
ENTITY_E2C = None
ENTITY_C2E = None

def get_lls_dic_from_response(response_ls, label_map_name=None):
    catego_ls = CATEGO_DICT[label_map_name]["short"]
    entity_lls = []
    temp_dic = {}
    for ind, response in enumerate(response_ls):
        entity_ls = []
        if 'result' in response:
            for catego in catego_ls:
                # import ipdb
                # ipdb.set_trace()
                if catego in response['result']:
                    cur_res = response['result'][catego].split(' ')
                    entity_ls.extend([catego+ATTACH_SYMBOL+ele for ele in cur_res])
        else:
            print(ind)
        entity_lls.append(entity_ls)

    for catego in catego_ls:
        temp_dic[catego] = []
        for ind, response in enumerate(response_ls):
            if catego in response['result']:
                cur_res = response['result'][catego].split(' ')
                temp_dic[catego].extend([str(ind)+'@#'+ele.upper() for ele in cur_res]) 
    return entity_lls, temp_dic

# SEP_AFTER_IND = '@#'
def entity_assign2catego(inp_entity_lls, catego_ls, sep_after_ind='', tup=False):
    ret_dic = dict(zip(catego_ls, [[[] for ii in range(len(inp_entity_lls))] for i in range(len(catego_ls))]))
    if tup:
        ret_dic = dict(zip(catego_ls, [([],[]) for _ in range(len(catego_ls))]))
    for ind, entity_ls in enumerate(inp_entity_lls):
        for entity in entity_ls:
            entity_split = entity.split(ATTACH_SYMBOL)
            if entity_split[0] in ["age", "sex", "bed_number"]:
                entity_split[1] = entity_split[1].replace("年龄", "").replace(":", "").replace("：", "").replace("岁", "").replace("性别", "").replace("床", "").replace("位", "").replace("号", "").replace("住", "").replace("病区", "").replace("病房", "")
            if len(entity_split) != 2:
                ipdb.set_trace()
            if entity_split[1]:
                if tup:
                    ret_dic[entity_split[0]][0].append(entity_split[1].upper())
                else:
                    ret_dic[entity_split[0]][ind].append(entity_split[1].upper()) # str(ind)+sep_after_ind+ # 注意必须+'@#'，不然对于同一个费用catego,可能出现第7个样本为10和而第71个样本为0，然后都变成710被append
    return ret_dic


pic_inds = []
def get_pic_truth(pic_label_dir, id_shared_ls=None): # pred
    if '.xlsx' in pic_label_dir:
        df = pd.read_excel(pic_label_dir)
        df = df.fillna("")
        for catego in ['票价', '燃油附加费', '民航发展基金', '其他税费', '总金额']:
            if catego in df:
                df[catego] = df[catego].map(lambda x: ('%.1f') % x)
        df['航班日期'] = df['航班日期'].dt.strftime('%Y-%m-%d')
        ret_dic = dict(zip(catego_ls, [[[] for ii in range(df.shape[0])] for i in range(len(catego_ls))]))
        for ind in range(df.shape[0]):
            # if df.loc[ind, "非出院记录"] != "":
            #     continue
            pic_inds.append(ind)
            for catego in catego_ls:
                pic_value = df.loc[ind, ENTITY_E2C[catego]]
                if pic_value:
                    pic_value = str(pic_value).replace('，', ',').replace('：', ':').replace(' ', '').replace('（', '(').replace('）', ')')
                    if catego == 'age':
                        pic_value = pic_value.replace('岁', '')
                    ret_dic[catego][ind].append(pic_value.upper())

        return ret_dic

    ret_dic = dict(zip(catego_ls, [[[] for ii in range(len(id_shared_ls))] for i in range(len(catego_ls))]))

    with open('/root/maojingwei579/entity_recognize_for_flight/output/boardcards/1500_288_augment/invalid_test_input.json', 'r') as rf:
        invalid_ls = json.load(rf)["for_all"]["multiple"]
    for ind, id_shared in enumerate(id_shared_ls):
        picture_correct_path = pic_label_dir+id_shared+'.correct'
        if os.path.exists(picture_correct_path) and id_shared not in invalid_ls:
            pic_inds.append(ind)
            with open(picture_correct_path, 'rb') as rf_picture:
                rf_picture_read = rf_picture.read()
                try:
                    rf_picture_read = rf_picture_read.decode(chardet.detect(rf_picture_read)['encoding'])
                except:
                    print(picture_correct_path)
            for line in rf_picture_read.split('\n'):
                if line != '':
                    line = line.replace('：', ':').strip()
                    try:
                        left, right = line.split(':')[0], line.split(':')[1]
                    except:
                        import ipdb; ipdb.set_trace()
                    if right:
                        right = right.replace(' ', '') # 去掉空格
                        if ',' in right and left=='航班号':
                            for right_splited in right.split(','):
                                ret_dic[ENTITY_C2E[left]][ind].append(right_splited.upper()) # str(ind)+SEP_AFTER_IND+
                        else: 
                            ret_dic[ENTITY_C2E[left]][ind].append(right.upper()) # str(ind)+SEP_AFTER_IND+
    return ret_dic
    # print(ret_dic)


PRED_TYPES = None
GOLD_TYPES = None
SAVE_DIR = None

overall_ls = ['overall_short', 'overall_long']
def update_num(inp_dic, to_set=True, is_test=None, id_shared=None, label_conflict=None):
    catego_short = catego_ls
    pred_num = dict(zip(PRED_TYPES, [dict(zip(catego_ls, [0 for ii in range(len(catego_ls))])) for i in range(len(PRED_TYPES))]))
    gold_num = dict(zip(GOLD_TYPES, [dict(zip(catego_ls, [0 for ii in range(len(catego_ls))])) for i in range(len(GOLD_TYPES))]))
    conflict_num, pic_num = [dict(zip(catego_ls, [0 for ii in range(len(catego_ls))])) for _ in range(2)]
    correct_num = {}
    for pred_type in PRED_TYPES:
        for gold_type in GOLD_TYPES:
            correct_num[pred_type+'_'+gold_type] = dict(zip(catego_ls, [0 for _ in range(len(catego_ls))]))
    def update_overall():
        overall_dic = dict(zip(overall_ls, [catego_short, []]))
        for overall in overall_ls:
            conflict_num[overall] = sum(conflict_num[catego] for catego in overall_dic[overall])
            pic_num[overall] = sum(pic_num[catego] for catego in overall_dic[overall])
            for pred_type in PRED_TYPES:
                pred_num[pred_type][overall] = sum([pred_num[pred_type][catego] for catego in overall_dic[overall]])
            for gold_type in GOLD_TYPES:
                gold_num[gold_type][overall] = sum([gold_num[gold_type][catego] for catego in overall_dic[overall]])
            for pred_type in PRED_TYPES:
                for gold_type in GOLD_TYPES:
                    combine_type = pred_type+'_'+gold_type
                    correct_num[combine_type][overall] = sum([correct_num[combine_type][catego] for catego in overall_dic[overall]])
    temp_wf = {}
    for typ in (PRED_TYPES+GOLD_TYPES+['textall', 'picall', 'diff']):
        temp_wf[typ] = open(os.path.join(SAVE_DIR, 'debug.'+typ), 'w', encoding='utf8')
        temp_wf[typ].write('*****'+typ.upper()+'*****\n')
    # 这里的 to set 只能去除同一个catego并且是同一个样本（由于加了样本编号）中的相同entity
    print(len(inp_dic['text'][catego_short[0]]))
    for ind in range(len(inp_dic['text'][catego_short[0]])):
        for catego in catego_short:
            # def write_func(typ):
                # tmp_set = set()
                # cur_ind = 0
                # for ele in inp_dic[typ][catego][ind]:
                #     if ele.split(SEP_AFTER_IND)[0] == str(cur_ind):
                #         tmp_set.add(ele.split(SEP_AFTER_IND)[1])
                #     else:
                #         temp_wf[typ].write('{}: {}\n'.format(cur_ind, tmp_set))
                #         tmp_set = set()
                #         while int(ele.split(SEP_AFTER_IND)[0])-cur_ind > 1:
                #             cur_ind += 1
                #             temp_wf[typ].write('{}: {}\n'.format(cur_ind, tmp_set))
                #         cur_ind += 1
                #         tmp_set.add(ele.split(SEP_AFTER_IND)[1])

            if 'pic' in inp_dic:
                # if id_shared in label_conflict[catego]:
                #     continue
                if ind in pic_inds:
                    pic_num[catego] += 1

                    if not set(inp_dic['pic'][catego][ind]):
                        # print(set(inp_dic['pic'][catego][ind]))
                        continue

                    if set(inp_dic['pic'][catego][ind]) != set(inp_dic['text'][catego][ind]):
                        continue

                #     if set(inp_dic['pic'][catego][ind]):
                #         temp_wf['textall'].write(catego+':'+'{}\n'.format(set(inp_dic['text'][catego][ind])))
                #         temp_wf['picall'].write(catego+':'+'{}\n'.format(set(inp_dic['pic'][catego][ind])))
                #         pic_num[catego] += 1
                #         if (set(inp_dic['pic'][catego][ind]) & set(inp_dic['text'][catego][ind])) == set():
                #             # temp_wf['diff'].write(catego+': '+'{}  ||  '+'{}\n'.format(set(inp_dic['text'][catego][ind]), set(inp_dic['pic'][catego][ind]))) # this produces unexpected result for unknown reasons
                #             temp_wf['diff'].write(catego+': '+'{} ||  '.format(set(inp_dic['text'][catego][ind])))
                #             temp_wf['diff'].write('{}\n'.format(set(inp_dic['pic'][catego][ind])))
                #             conflict_num[catego] += 1
                #             continue    
                else:
                    continue

            
            for pred_type in PRED_TYPES:
                # print(pred_type, catego, ind)
                pred_num[pred_type][catego] += len(set(inp_dic[pred_type][catego][ind])) if to_set else len(inp_dic[pred_type][catego][ind])
                if temp_wf:
                    temp_wf[pred_type].write(str(ind+1)+' '+ENTITY_E2C[catego]+':'+'{}\n'.format(set(inp_dic[pred_type][catego][ind])))
            for gold_type in GOLD_TYPES:
                gold_num[gold_type][catego] += len(set(inp_dic[gold_type][catego][ind])) if to_set else len(inp_dic[gold_type][catego][ind])
                if temp_wf:
                    temp_wf[gold_type].write(str(ind+1)+' '+ENTITY_E2C[catego]+':'+'{}\n'.format(set(inp_dic[gold_type][catego][ind])))
            for pred_type in PRED_TYPES:
                for gold_type in GOLD_TYPES:
                    combine_type = pred_type+'_'+gold_type
                    correct_num[combine_type][catego] += len(set(inp_dic[pred_type][catego][ind])&set(inp_dic[gold_type][catego][ind]))

        if temp_wf:
            for typ in (PRED_TYPES+GOLD_TYPES+['diff']):
                temp_wf[typ].write('\n')
                temp_wf[typ].flush()
                # ipdb.set_trace()

    for v in temp_wf.values(): 
        v.close()
    update_overall()
    if 'pic' in inp_dic:
        for catego in catego_short+['overall_short']:
            print('{}: {}, {}, {:.4f}'.format(catego, conflict_num[catego], pic_num[catego], conflict_num[catego]/pic_num[catego]), end='  ')
        print('\n')
    return pred_num, gold_num, correct_num
                

def cal_p_r_f1(pred_num, gold_num, correct_num, csv_file=None):
    p, r, f1 = [dict() for _ in range(3)]
    if csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['字段类别', '预测正确个数', '预测总数', '真实总数', '准确率', '召回率', 'f1'])
    for pred_type in PRED_TYPES:
        for gold_type in GOLD_TYPES:
            print('######### {} || {} #########'.format(pred_type, gold_type))
            for catego in (catego_ls+overall_ls):
                p[catego] = correct_num[pred_type+'_'+gold_type][catego] / (pred_num[pred_type][catego]+1)
                r[catego] = (correct_num[pred_type+'_'+gold_type][catego]+1) / (gold_num[gold_type][catego]+1)
                f1[catego] = 2*p[catego]*r[catego] / (p[catego]+r[catego])
#                 write_print("category: {:20}\t correct: {:<5}\t pred: {:<5}\t gold: {:<5}\t f1: {:.4f}".format(catego, correct_num[pred_type+'_'+gold_type][catego], pred_num[pred_type][catego], gold_num[gold_type][catego], f1[catego]), wf=logfile)
                print("category: {:20}\t correct: {:<5}\t pred: {:<5}\t gold: {:<5}\t precision: {:.4f}\t f1: {:.4f}".format(catego, correct_num[pred_type+'_'+gold_type][catego], pred_num[pred_type][catego], gold_num[gold_type][catego], p[catego], f1[catego]))
                if csv_file:
                    csv_writer.writerow([ENTITY_E2C.get(catego, "综合"), correct_num[pred_type+'_'+gold_type][catego], pred_num[pred_type][catego], gold_num[gold_type][catego], '{:.3f}'.format(p[catego]), '{:.3f}'.format(r[catego]), '{:.3f}'.format(f1[catego])])
            if gold_type == 'text' and pred_type == 'model':
                f1_ret = f1['overall_short']
    if csv_file:
        csv_file.close()

#     return f1_ret



def evaluate(truth_path=None, pred=None, label_map_name=None, id_shared_ls=None):
    # import ipdb
    # ipdb.set_trace()
    temp_dic = {}
    global catego_ls, GOLD_TYPES, PRED_TYPES, ENTITY_C2E, ENTITY_E2C, SAVE_DIR
    SAVE_DIR = './'
    PRED_TYPES = ['pred']
    GOLD_TYPES = ['text']
    ENTITY_C2E = ENTITY_C2E_DICT[label_map_name]
    ENTITY_E2C = ENTITY_E2C_DICT[label_map_name]
    catego_ls = CATEGO_DICT[label_map_name]['short'] + CATEGO_DICT[label_map_name]['long']
    if isinstance(pred, str):
        (pred_entity_lls, pred_label_lls, pred_text_lls) = extract_entity(pred)
        SAVE_DIR = pred.replace('test_predictions.txt', '')
    elif isinstance(pred, list):
        pred_entity_lls = pred

    if truth_path:
        (true_entity_lls, true_label_lls, true_text_lls) = extract_entity(truth_path)
        assert len(pred_entity_lls) == len(true_entity_lls)
        get_num(true_entity_lls, pred_entity_lls)
        get_num(true_entity_lls, pred_entity_lls, False)
        temp_dic['text'] = entity_assign2catego(true_entity_lls, catego_ls)

    if isinstance(pred, str):
        label_accuracy_ls = []
        pred_label = []
        true_label = []
        for true_label_ls, pred_label_ls in zip(true_label_lls, pred_label_lls):
            l_pred_label_ls = len(pred_label_ls)
            label_accuracy_ls.append((l_pred_label_ls, accuracy_score(true_label_ls[:l_pred_label_ls], pred_label_ls)))
            pred_label.extend(pred_label_ls)
            true_label.extend(true_label_ls[:l_pred_label_ls])
        print(sum([a[1] for a in label_accuracy_ls])/len(label_accuracy_ls))
        print(classification_report(true_label, pred_label))

    # with open('text.csv', 'w', encoding='gb2312') as wf:
    #     csv_writer = csv.writer(wf)
    #     headers = list(temp_dic['text'].keys())
    #     csv_writer.writerow([ENTITY_E2C_DICT[label_map_name][header] for header in headers])
    #     length = len(temp_dic['text'][headers[0]])
    #     for i in range(length):
    #         csv_writer.writerow(['@#'.join(temp_dic['text'][header][i]) for header in headers])
    temp_dic['pred'] = entity_assign2catego(pred_entity_lls, catego_ls)
    # with open('pred.csv', 'w', encoding='gb2312') as wf:
    #     csv_writer = csv.writer(wf)
    #     headers = list(temp_dic['text'].keys())
    #     csv_writer.writerow([ENTITY_E2C_DICT[label_map_name][header] for header in headers])
    #     length = len(temp_dic['pred'][headers[0]])
    #     for i in range(length):
    #         csv_writer.writerow(['@#'.join(temp_dic['pred'][header][i]) for header in headers])

    if isinstance(pred, str):
        PRED_TYPES += ['pred_rule']
        temp_dic['pred_rule'] = dict(zip(catego_ls, [[[] for ii in range(len(true_entity_lls))] for i in range(len(catego_ls))]))
        for catego in catego_ls:
            for ind, value_ls in enumerate(temp_dic['pred'][catego]):
                # if ind == 142 and catego == "Destination_PY":
                #     ipdb.set_trace()
                for value in value_ls:
                    value = post_func(value.upper(), ''.join(pred_text_lls[ind]), 'interface', catego, label_map_name=label_map_name)
                    # print(value)
                    if value:
                        if value not in temp_dic['pred_rule'][catego][ind]:
                            temp_dic['pred_rule'][catego][ind].append(value)
                if catego in NEED_POSTPROCESS and len(temp_dic['pred_rule'][catego][ind]) > 1:
                    # ipdb.set_trace()
                    temp_dic['pred_rule'][catego][ind] = postprocess_ls(temp_dic['pred_rule'][catego][ind])


    # with open('pred_rule.csv', 'w', encoding='gb2312') as wf:
    #     csv_writer = csv.writer(wf)
    #     headers = list(temp_dic['pred_rule'].keys())
    #     csv_writer.writerow(headers)
    #     length = len(temp_dic['pred_rule'][headers[0]])
    #     for i in range(length):
    #         csv_writer.writerow(['@#'.join(temp_dic['pred_rule'][header][i]) for header in headers])
    # if id_shared_ls or (label_map_name == "discharge_records.json" and isinstance(pred, list)):
    if PATH_DICT[label_map_name]['pic_label'] and id_shared_ls:
        # import ipdb
        # ipdb.set_trace()
        temp_dic['pic'] = get_pic_truth(PATH_DICT[label_map_name]['pic_label'], id_shared_ls)
        GOLD_TYPES += ['pic']

    # if 'text' not in temp_dic:
    #     temp_dic['text'] = copy.deepcopy(temp_dic['pic'])

    row = ["文件路径"]

    if 'pic' in temp_dic and 'text' in temp_dic:
        print('hhhhhhh')
        notgb2312_ls = ['\u3007', '\u2022', '\xd5', '\xcf', '\xc1', '\u77f0', '\xc4']
        with open(os.path.join(SAVE_DIR, 'pred_text_pic.csv'), 'w', encoding='gb2312') as wf:
            csv_writer = csv.writer(wf)
            headers = list(temp_dic['text'].keys()) # [:9]
            zh_headers_pred = [ENTITY_E2C[header] + '预测值' for header in headers]
            zh_headers_text = [ENTITY_E2C[header] + '文本标签' for header in headers]
            zh_headers_pic = [ENTITY_E2C[header] + '图片标签' for header in headers]
            # csv_writer.writerow([ENTITY_E2C_DICT[label_map_name][header] for header in headers])
            length = len(temp_dic['text'][headers[0]])

            for ele in zip(zh_headers_pred, zh_headers_text, zh_headers_pic):
                row.extend(ele)
            print(row)
            csv_writer.writerow(row)
            # for header in headers:
            #     print(len(temp_dic['pred'][header]), len(temp_dic['text'][header]), len(temp_dic['pic'][header]))
            for i in range(length):
                row = [id_shared_ls[i]]
                for ele in zip(
                    ['@#'.join(temp_dic['pred'][header][i]) for header in headers], ['@#'.join(temp_dic['text'][header][i]) for header in headers], ['@#'.join(temp_dic['pic'][header][i]) for header in headers]):
                    list_ele = []
                    for subele in ele:
                        for notgb2312 in notgb2312_ls:
                            subele = subele.replace(notgb2312, '')
                        list_ele.append(subele)
                    row.extend(list_ele)

                # print(row)
                csv_writer.writerow(row)

    elif 'pic' in temp_dic:
        notgb2312_ls = ['\u3007', '\u2022', '\xd5', '\xcf', '\xc1', '\u77f0', '\xc4']
        with open(os.path.join(SAVE_DIR, 'pred_pic.csv'), 'w', encoding='gb2312') as wf:
            csv_writer = csv.writer(wf)
            headers = list(temp_dic['pic'].keys()) # [:9]
            zh_headers_pred = [ENTITY_E2C[header]+'预测值' for header in headers]
            zh_headers_pic = [ENTITY_E2C[header] + '图片标签' for header in headers]
            length = len(temp_dic['pic'][headers[0]])

            for ele in zip(zh_headers_pred, zh_headers_pic, ['备注' for header in headers]):
                row.extend(ele)
            print(row)
            csv_writer.writerow(row)
            for i in range(length):
                row, tmp_ls = [[] for _ in range(2)]
                for header in headers:
                    if temp_dic['pred'][header][i] != temp_dic['pic'][header][i]:
                        # if i == 10 and header == 'Oil_Fee':
                        #     import ipdb
                        #     ipdb.set_trace()
                        wrong_type = '错误-抽取错误'
                        if temp_dic['pic'][header][i]:
                            if temp_dic['pic'][header][i][0] not in aa[i]:
                                wrong_type = '错误-OCR错误'
                        tmp_ls.append(wrong_type)
                    else:
                        tmp_ls.append('')
                for ele in zip(
                        ['@#'.join(temp_dic['pred'][header][i]) for header in headers],
                        ['@#'.join(temp_dic['pic'][header][i]) for header in headers], tmp_ls):
                    list_ele = []
                    for subele in ele:
                        for notgb2312 in notgb2312_ls:
                            subele = subele.replace(notgb2312, '')
                        list_ele.append(subele)
                    row.extend(list_ele)

                # print(row)
                csv_writer.writerow(row)

        temp_dic['text'] = copy.deepcopy(temp_dic['pic'])


    elif 'text' in temp_dic:
        row = []
        notgb2312_ls = ['\u3007', '\u2022', '\xd5', '\xcf', '\xc1', '\u77f0', '\xc4']
        with open(os.path.join(SAVE_DIR, 'pred_text.csv'), 'w', encoding='gb2312') as wf:
            csv_writer = csv.writer(wf)
            headers = list(temp_dic['text'].keys())[:9]
            zh_headers_pred = [ENTITY_E2C[header] + '预测值' for header in headers]
            zh_headers_text = [ENTITY_E2C[header] + '文本标签' for header in headers]
            # csv_writer.writerow([ENTITY_E2C_DICT[label_map_name][header] for header in headers])
            length = len(temp_dic['text'][headers[0]])

            for ele in zip(zh_headers_pred, zh_headers_text):
                row.extend(ele)
            csv_writer.writerow(row)
            for i in range(length):
                row = []
                for ele in zip(
                    ['@#'.join(temp_dic['pred'][header][i]) for header in headers], ['@#'.join(temp_dic['text'][header][i]) for header in headers]):
                    list_ele = []
                    for subele in ele:
                        for notgb2312 in notgb2312_ls:
                            subele = subele.replace(notgb2312, '')
                        list_ele.append(subele)
                    row.extend(list_ele)

                # print(row)
                csv_writer.writerow(row)

    
    pred_num, gold_num, correct_num = update_num(inp_dic=temp_dic)
    csv_w = open(os.path.join(SAVE_DIR, 'tmp.csv'), 'w', encoding='gb2312')
    cal_p_r_f1(pred_num, gold_num, correct_num, csv_w)
    # del pred_num
    # del correct_num
    # del gold_num

    # update_num(inp_dic=temp_dic, to_set=False)
    # update_overall()
    # cal_p_r_f1()

def temp_evaluate(truth_path, pred, label_map_name=None):
    global catego_ls, GOLD_TYPES, PRED_TYPES
    PRED_TYPES = ['pred']
    GOLD_TYPES = ['text']
    (true_entity_lls, true_label_lls, true_text_lls) = extract_entity(truth_path)
    catego_ls = CATEGO_DICT[label_map_name]['short']+CATEGO_DICT[label_map_name]['long']
    temp_dic = {}
    temp_dic['text'] = entity_assign2catego(true_entity_lls, catego_ls)
    temp_dic['pred'] = pred
    pred_num, gold_num, correct_num = update_num(inp_dic=temp_dic) 
    cal_p_r_f1(pred_num, gold_num, correct_num)
