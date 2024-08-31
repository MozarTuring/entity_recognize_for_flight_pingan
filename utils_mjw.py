import re
import numpy as np
from constants_jw import PRD
import json
import pandas as pd
import ipdb
import jieba
from zhon.hanzi import punctuation
import requests, time


df = pd.read_excel('/data/maojingwei/entity_recognize_for_flight/hospital_ICD10/hospital_ICD10.xlsx')
hospital_list = list(set(df.loc[:, "医院名称"].fillna('').tolist()))
disease_list = list(set(df.loc[:, "疾病名称"].fillna('').tolist()))
disease_list_cut = [list(jieba.cut(ele)) for ele in disease_list]

with open('/data/maojingwei/entity_recognize_for_flight/pinyin.txt', 'r', encoding='utf8') as rf:
    rf_read = rf.read()
pinyinLib = rf_read.split('\n')

def pinyin_or_word(string):
    '''
    judge a string is a pinyin or not.
    pinyinLib comes from a txt file.
    '''
    
    max_len = 6   # 拼音最长为6
    string = string.lower()
    stringlen = len(string)
    result = []
    while True:
        matched = 0
        matched_word = ''
        if stringlen < max_len:
            max_len = stringlen                
        for i in range(max_len+1, 1, -1):
            s = string[:i]
            if s in pinyinLib:
                matched_word = s
                matched = i
                break
        if len(matched_word) == 0:
            break
        else:
            result.append(s)
            string = string[len(s):]
            stringlen = len(string)
            if stringlen == 0:
                break
    return result


# import ipdb; ipdb.set_trace()
# pinyin_or_word('kailin')

with open('/data/maojingwei/entity_recognize_for_flight/loc_pys.json', 'r', encoding='utf8') as rf:
    LOC_PYS = json.load(rf)

# LOC_PYS = ['PHUKET', 'TAIPEI', 'GUANGZHOU', 'PUDONG', 'DALIAN', 'WUHAN', 'LANZHOU', 'KUNMING', 'XIAMEN', 'SHENZHEN', 'HOHHOT', 'FOSHAN', 'BEIHAI', 'GUIYANG', 'CHANGSHA', 'SHIJIAZHUANG', 'CHIANGMAI', 'CHENGDU', 'LIUPANSHUI', 'CHONGQING', 'BEIJING', 'WENZHOU', 'NANNING'] + ['SHANGHAI', 'BANGKOK'] #  'SHENNONGJI', 'SHENNONGJIA',

MUST_BE_START = ['L', 'T', 'F', 'H']
IMPOSSIBLE_2GRAM = []
for ele1 in MUST_BE_START:
    for ele2 in MUST_BE_START:
        IMPOSSIBLE_2GRAM.append(ele1+ele2)
IMPOSSIBLE_END_OF_ENGLISH = ['ZHA', 'ZH']

MONTHS_EN = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
MONTHS_NUMBER = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
EN2NUMBER = dict(zip(MONTHS_EN, MONTHS_NUMBER))

NEED_POSTPROCESS = ['Flight_Number', 'Name_PY', "Flight_Date"]

FLIGHT_NUMBER_DICT = {"东航":"MU", "南航":"CZ", "厦航":"MF", "海航":"HU"}

def post_func(value, tokens_returned, pred_type='model', catego=None, label_map_name=None):

    # if isinstance(tokens_returned, list):
    #     inp_str = ''.join(tokens_returned[ind_batch]).upper()
    assert isinstance(tokens_returned, str)
    inp_str = tokens_returned.upper() 
    inp_str = inp_str.replace('^', '')
    start_pos = inp_str.find(value)
    end_pos = start_pos + len(value)
    
    def post_name(value):
        if "姓名" in inp_str:
            if "姓名" not in inp_str[max(0,start_pos-50):min(len(inp_str), end_pos+50)]:
                return ''
        return value

    def post_xingchengdan_name(value):
        if len(value) == 1:
            return ''
        value = re.sub(r'\(.+\)', '', value)
        if re.search(r'[a-zA-Z]', value):
            value = post_namepy(value)
        return value
    
    def post_department(value):
        # if '科' in value:
        #     value = value[:value.find('科')+1]
        
        if inp_str[end_pos] in [')', '）']:
            value = value + ')'
        if inp_str[end_pos:end_pos+2] == '住院' and inp_str[end_pos:end_pos+3] != '住院号':
            value = value + '住院'
        value = value.replace('料', '科') # .replace('病区', '')
        return value

    def post_date(value):
        pat = re.compile(r'[\d\-\.\/:：,\^年月日时分秒点]+')
        res = pat.findall(value)
        if res:
            value = res[0]
        if len(value) <= 2:
            return ''
        return value

    def post_adnumber(value):
        pat = re.compile(r'[\da-zA-Z\+\-]+')
        res = pat.findall(value)
        if res:
            value = res[0]
        return value
    
    def post_hosname(value):
        if len(value) <= 1:
            return ''
        return value

    def post_flight_date(value):
        value = value.replace(' ', '').replace('O', '0').replace('I', '1').replace('T', '1').replace('Q', '0')
        value = value.replace('0C1', 'OCT').replace('N0V', 'NOV').replace('0CT', 'OCT')
        pat = re.compile(r'[a-zA-Z]')
        if pat.findall(value):
            for ele in MONTHS_EN:
                if ele in value:
                    return value
            return ''
        return value
    
    def post_flight_date_prd(value):
        # value = value.replace('0CT', 'OCT').replace(' ', '')
        value = re.sub(r'[ :：\.]', '-', value)
        value.replace('.', '-')
        for ele in MONTHS_EN:
            if ele in value:
                pos_start = value.find(ele)
                if value[pos_start+3:]:
                    value = value[pos_start+3:]+'-'+EN2NUMBER[ele]+'-'+value[:pos_start]
                else:
                    value = EN2NUMBER[ele]+'-'+value[:pos_start]
                break
        return value

    def post_flight_number(value):
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if re.search(pattern, value):
            return None
        number = ''.join(re.findall(r'\d', value))
        for k in FLIGHT_NUMBER_DICT.keys():
            temp = FLIGHT_NUMBER_DICT[k]+number
            if k in inp_str and temp in inp_str:
                return temp

        return value

    def post_namepy(value):
        start_pos = inp_str.find(value)
        end_pos = start_pos + len(value)

        for imp_end_eng in IMPOSSIBLE_END_OF_ENGLISH:
            if inp_str[start_pos-len(imp_end_eng):start_pos] == imp_end_eng:
                start_pos -= len(imp_end_eng)
        value = inp_str[start_pos:end_pos]

        res_search = re.search('MS|MR', inp_str[end_pos:end_pos+4])
        if res_search:
            end_pos = end_pos+res_search.span()[1]
            value = inp_str[start_pos:end_pos]
        try:
            if value[-1] == 'M':
                value = inp_str[start_pos:end_pos+1]
        except:
            import ipdb; ipdb.set_trace()
        if value[0] == 'U':
            value = inp_str[start_pos-1:end_pos]
        min_cut = len(value)
        for im_2gram in IMPOSSIBLE_2GRAM:
            if im_2gram in value:
                cut = value.find(im_2gram)
                if cut < min_cut:
                    min_cut = cut
        if min_cut == 0:    
            value = value[2:]
        else:
            value = value[:min_cut]
        value = value.replace('NAME', '')
        pat = 'MS|MR'
        res_search = re.search(pat, value)
        if res_search:
            value = value[:res_search.span()[1]]
        pat = re.compile(r'[\u4e00-\u9fa5]')
        res_search = re.search(pat, value)
        if res_search:
            value = value[:res_search.span()[0]]
        if '[CLS]' in value:
            import ipdb; ipdb.set_trace()
        if len(value) == 1:
            return None

        if 'MR' not in value and 'MS' not in value:
            if '/' in value:
                r_pinyin_l = pinyin_or_word(value.split('/')[0])
                r_pinyin_r = pinyin_or_word(value.split('/')[1])
                if ''.join(r_pinyin_r) != value.split('/')[1].lower():
                    if 'HE/GA' in value:
                        import ipdb; ipdb.set_trace()
                    if not r_pinyin_r:
                        pass
                    elif r_pinyin_r[0][0] == value.split('/')[1][0].lower():
                        value = value.split('/')[0] + '/' + r_pinyin_r[0].upper()
                elif len(r_pinyin_r) > 2:
                    value = value.split('/')[0] + '/' + ''.join(r_pinyin_r[:2]).upper()
            else:
                r_pinyin_l = pinyin_or_word(value[:7])
        return value.replace('+', '')
    
    def post_locpy(value):
        if len(value) <= 3 or 'GATE' in value or 'PLEASE' in value:
            return None
        
        if value in LOC_PYS:
            return value

        flag = False
        for loc_py in LOC_PYS:
            if loc_py in value:
                if len(loc_py) / len(value) >= 0.6:
                    value = loc_py
                    flag = True
                    break
            elif value in loc_py:
                if len(value) / len(loc_py) >= 0.5:
                    if loc_py not in inp_str:
                        continue
                    value = loc_py
                    flag = True
                    break

        if flag:
            return value
        else:
            return None
        
        # if 'DEPART' in value:
        #     return None
    
    if catego == 'Flight_Number':
        value = value.replace(' ','')[:6]
    
    if catego == "Flight_Date":
        value = post_flight_date(value)
        if PRD and value:
            value = post_flight_date_prd(value)

    if label_map_name == 'boardcards.json':
        if len(value) <= 1:
            return None                 

        if catego == 'Name_PY':
            if pred_type == 'interface':
                return value.replace('+', '')
            return post_namepy(value)
        elif catego in ['Destination_PY', 'Origin_PY']:
            return post_locpy(value)
        elif catego == 'Flight_Number':
            return post_flight_number(value)
        elif catego == 'Name':
            # import ipdb; ipdb.set_trace()
            return post_name(value)
        else:
            return value
    elif label_map_name == 'xingchengdan.json':
        if 'Fee' in catego:
            value = re.sub('[a-zA-Z]', '', value)
        return value
    elif label_map_name == 'xingchengdan_3code.json':
        if catego == "Flight_Number":
            return post_flight_number(value)
        elif catego == "Name":
            return post_xingchengdan_name(value)
        return value
    elif label_map_name == 'discharge_records.json':
        if 'Date' in catego:
            return post_date(value)
        if catego == 'Admission_Number':
            return post_adnumber(value)
        if catego == 'Hospital_Name':
            return post_hosname(value)
        if catego == 'Department':
            return post_department(value)
        # if not PRD:
        #     if catego == 'Department':
        #         return post_department(value)
        return value

    elif label_map_name == 'imaging_report.json':
        return value
    
    else:
        import ipdb; ipdb.set_trace()


def extractFee_from_ocr_result(ocr_result, inp_dic):
    fees={'rule1':[], 'rule2':[]}
    coord_x = {}
    for ind, ele in enumerate(ocr_result['result']['info']):
        if ('.0' in ele['text'] or '0.' in ele['text'] or '.o' in ele['text'].lower() or 'o.' in ele['text'].lower()) and not re.search(r'[\u4e00-\u9fa5]', ele['text']) and 'e-' not in ele['text'].lower():
            fees['rule1'].append(ele)
        if len(set('燃油附加费')&set(ele['text'])) >= 2:
            coord_x['燃油'] = ele['coord'][0]['x']
            oil_ind = ind
            oil_fee = ele
            oil_x = oil_fee['coord'][0]['x']
        if len(set('民航发展基金')&set(ele['text'])) >= 2:
            coord_x['民航'] = ele['coord'][0]['x']
        if len(set('合计')&set(ele['text'])) == 2:
            coord_x['合计'] = ele['coord'][0]['x']
        if 'cn' in ele['text'].lower():
            fees['rule2'].append(ele)
    if len(coord_x) != 3:
        print('can not extract fee from ocr result')
        return None

    if len(fees['rule1']) == 5:
        fees['rule1'].sort(key=lambda x: x['coord'][0]['x'])
        count = 0
        for catego in ['Ticket_Fee', 'Develop_Fee', 'Oil_Fee', 'Other_Fee', 'Total_Fee']:
            inp_dic[catego].append(re.sub('[a-zA-Z ]', '', fees['rule1'][count]['text']))
            count += 1
    else:
        left_fees = []
        for fee in fees['rule1']:
            cur_fee_x = fee['coord'][0]['x']
            if coord_x['民航'] < cur_fee_x < coord_x['燃油']:
                inp_dic['Develop_Fee'].append(re.sub('[a-zA-Z ]', '', fee['text']))
            elif cur_fee_x < coord_x['民航']:
                inp_dic['Ticket_Fee'].append(re.sub('[a-zA-Z ]', '', fee['text']))
            elif cur_fee_x > coord_x['合计']:
                inp_dic['Total_Fee'].append(re.sub('[a-zA-Z ]', '', fee['text']))
            else:
                left_fees.append(fee)
        for fee in left_fees:
            cur_fee_x = fee['coord'][0]['x']
            if cur_fee_x-coord_x['燃油'] < coord_x['合计']-cur_fee_x:
                inp_dic['Oil_Fee'].append(re.sub('[a-zA-Z ]', '', fee['text']))
            else:
                inp_dic['Other_Fee'].append(re.sub('[a-zA-Z ]', '', fee['text']))


def match_with_standard(inp_dic):
    for k in ["dis_diag", "ad_diag"]:
        for ele in inp_dic[k]:
            ele = re.sub(r"[%s]+" %punctuation, "", ele)
            ele = re.sub(r"[\(\):]", "", ele)
            print(set(jieba.cut(ele)))
            overlap_ls = [(''.join(disease), len(set(disease)&set(jieba.cut(ele)))) for disease in disease_list_cut]
            overlap_ls.sort(key=lambda x:x[1])
            candidate_ls = overlap_ls[-10:]
            print(candidate_ls)
            for candidate in candidate_ls:
                if candidate[0] in ele:
                    print(candidate)




def postprocess_dic(inp_dic):
    for k in NEED_POSTPROCESS:
        if k in inp_dic:
            length = len(inp_dic[k])
            if length > 1:
                ls = []
                for i in range(length):
                    cur_ele = inp_dic[k][i]
                    flag = False
                    for j in range(length):
                        if j != i:
                            if cur_ele in inp_dic[k][j]:
                                flag = True
                                break
                    if not flag:
                        ls.append(cur_ele)
                inp_dic[k] = ls


def postprocess_ls(inp_ls):
    length = len(inp_ls)
    assert length > 1
    ls = []
    for i in range(length):
        cur_ele = inp_ls[i]
        flag = False
        for j in range(length):
            if j != i:
                if cur_ele in inp_ls[j]:
                    flag = True
                    break
        if not flag:
            ls.append(cur_ele)
    return ls


def add_zero(inp):
    if len(inp) == 1:
        return "0"+inp
    return inp

def post_discharge_records(value, catego):
    if catego in ["Discharge_Date", "Admission_Date"]:
        res = re.sub(r'[年月日]', '-', value)
        res = re.sub(r'\.', '-', value)
        res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', res)
        if res:
            value = res[0]
            year, month, day = value.split('-')
            value = "-".join([year, add_zero(month), add_zero(day)])

    # value = value.replace('mjwemptymjw', '\n').replace('MJWEMPTYMJW', '\n')
    return value

def post_imaging_report(value, catego):
    if catego in ["Check_Date", "Report_Date"]:
        value = re.sub(r'\d\d:\d\d:\d\d', '', value)
        value = re.sub(r'[年月日\.\/]', '-', value)
        # res = re.sub(r'\.', '-', value)
        res = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', value)
        if res:
            value = res[0]
            year, month, day = value.split('-')
            value = "-".join([year, add_zero(month), add_zero(day)])

    # value = value.replace('mjwemptymjw', '\n').replace('MJWEMPTYMJW', '\n')
    return value

if __name__ == "__main__":
    tmp_dic = {}
    # tmp_dic["dis_diag"] = ["西医诊断:左胫骨平台骨折(SCHATZKERII中医诊断:损伤骨折型)气滞血瘀证"]
    # match_with_standard(tmp_dic)


    # saved_dic = {}
    # i = 0
    # start = time.time()
    # batch_size = 64
    # while True:
    #     data_sent = {"id":"123", "texts":disease_list[i*batch_size:(i+1)*batch_size], "is_tokenized":False}
    #     print((i+1)*batch_size, disease_list[i*batch_size:(i+1)*batch_size])
    #
    #     while True:
    #         try:
    #             response = requests.post("http://mjw-bert-as-service-gpu-stg.icore-aas-rbt-gpu-stg.paic.com.cn/encode", data=json.dumps(data_sent), headers={"Content-Type": "application/json"})
    #         except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
    #             if "10054" in str(e) or "104" in str(e):
    #                 continue
    #             else:
    #                 raise Exception(e)
    #         break
    #     response_dic = json.loads(response.text)
    #     saved_dic.update(dict(zip(disease_list[i*batch_size:(i+1)*batch_size], response_dic["result"])))
    #     if (i+1)*batch_size >= len(disease_list):
    #         break
    #     i += 1
    #
    # print(len(saved_dic))
    # print(time.time()-start)
    # with open("hospital_ICD10/bert_embeddings.json", "w", encoding='utf8') as wf:
    #     json.dump(saved_dic, wf)

