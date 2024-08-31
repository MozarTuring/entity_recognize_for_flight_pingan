import re, os

def get_exchange_dict(inp_dic):
    ret_dic = {}
    for k, v in inp_dic.items():
        ret_dic[v] = k
    return ret_dic

CATEGO_DICT = {
        "discharge_records.json":{
            "short":['Discharge_Date', 'Admission_Number', 'Hospital_Name', 'Admission_Date', 'Department', "Name", "sex", "age", "bed_number"], 
            "long":['ad_diag', 'dis_situ', 'dis_diag', "dis_note", "ad_main"]}, 
        "boardcards.json":{
            "short":['Flight_Date', 'Flight_Number', 'Name', 'Name_PY', 'Destination', 'Destination_PY', 'Origin', 'Origin_PY'], 
            "long":[]},
        "xingchengdan.json":{
            "short":['Name', 'Flight_Number', 'Flight_Date', 'Origin', 'Destination', 'Ticket_Fee', 'Develop_Fee', 'Oil_Fee', 'Other_Fee', 'Total_Fee'],
            "long":[]
        },
        "xingchengdan_3code.json":{
            "short":['Name', 'Flight_Number', 'Flight_Date', 'Origin', 'Destination', 'Ticket_Fee', 'Develop_Fee', 'Oil_Fee', 'Other_Fee', 'Total_Fee', "Origin_Code", "Destination_Code"],
            "long":[]
        },
        "imaging_report.json":{
            "short":['Name', 'Hospital_Name', 'Check_Date', 'Check_Part', 'Report_Date', 'Description', 'Conclusion', 'Check_Type'],
            "long":[]
        },
        "disability_report.json":{
            "short":["Institute_Name", "Institute_Number", "Institute_Report_Number", "Client", "Application_Reason", "Accept_Date", "Assess_Date", "Assess_Place", "Present_People", "Assessed_Person", "Assess_Equipment", "Body_Check", "Disabled_Part", "Disabled_Side_Angle", "Healthy_Side_Angle", "Loss_Percentage", "Disabled_Level", "Following_Fee", "Assess_Metric", "Assess_Clause", "Interrupt_Duration", "Nurse_Duration", "Nutrition_Duration", "Assess_Person1", "Assess_Person1_Number", "Assess_Person2", "Assess_Person2_Number", "Assess_Person3", "Assess_Person3_Number", "Assess_Report_Date", "Assessed_Person_Number"],
            "long":[]
        }
            }

ENTITY_C2E_DICT = {
    "boardcards.json":{
        '航班号':'Flight_Number', '航班日期':'Flight_Date', '姓名':'Name', '姓名拼音':'Name_PY', '出发地':'Origin', '出发地拼音':'Origin_PY', '目的地':'Destination', '目的地拼音':'Destination_PY'},
    "xingchengdan.json":{
        '姓名':'Name', '航班号':'Flight_Number', '航班日期':'Flight_Date', '出发地':'Origin', '目的地':'Destination', '票价':'Ticket_Fee', '民航发展基金':'Develop_Fee', '燃油附加费':'Oil_Fee', '其他税费':'Other_Fee', '总金额':'Total_Fee'},
    "xingchengdan_3code.json":{
        '姓名':'Name', '航班号':'Flight_Number', '航班日期':'Flight_Date', '出发地':'Origin', '目的地':'Destination', '票价':'Ticket_Fee', '民航发展基金':'Develop_Fee', '燃油附加费':'Oil_Fee', '其他税费':'Other_Fee', '总金额':'Total_Fee', '出发地三字码':"Origin_Code", "目的地三字码":"Destination_Code"},
    "discharge_records.json":{
        "出院日期":'Discharge_Date', "住院号":'Admission_Number',  "医院名称":'Hospital_Name', "入院日期":'Admission_Date', "科室":'Department', "伤者姓名":"Name", "性别": "sex", "年龄": "age", "床号": "bed_number", "入院诊断":'ad_diag', "出院情况":'dis_situ', "出院诊断":'dis_diag', "出院医嘱": "dis_note", "入院主诉":"ad_main"},
    "imaging_report.json":{
        "伤者姓名":'Name', "检查报告医院名称":"Hospital_Name", "检查日期":'Check_Date', "检查部位/项目":'Check_Part', "检查报告出具时间":'Report_Date', "检查报告对检查部位的描述":'Description', "诊断（结论）":"Conclusion", "检查类型":'Check_Type'
    },
    "disability_report.json":dict(
        zip(["鉴定所名称", "鉴定所许可证号", "鉴定意见书编号", "委托方", "委托事项", "受理日期", "鉴定日期", "鉴定地点", "在场人员", "被鉴定人", "检验设备", "体格检查", "评残部位", "患侧活动度", "健侧活动度", "活动度丧失", "伤残等级", "后续治疗费", "评残标准", "评残条款", "误工期", "护理期", "营养期", "鉴定人1", "鉴定人1执业证号", "鉴定人2", "鉴定人2执业证号", "鉴定人3", "鉴定人3执业证号", "鉴定报告出具日", "被鉴定人身份证号码"], ["Institute_Name", "Institute_Number", "Institute_Report_Number", "Client", "Application_Reason", "Accept_Date", "Assess_Date", "Assess_Place", "Present_People", "Assessed_Person", "Assess_Equipment", "Body_Check", "Disabled_Part", "Disabled_Side_Angle", "Healthy_Side_Angle", "Loss_Percentage", "Disabled_Level", "Following_Fee", "Assess_Metric", "Assess_Clause", "Interrupt_Duration", "Nurse_Duration", "Nutrition_Duration", "Assess_Person1", "Assess_Person1_Number", "Assess_Person2", "Assess_Person2_Number", "Assess_Person3", "Assess_Person3_Number", "Assess_Report_Date", "Assessed_Person_Number"]))
}


ENTITY_E2C_DICT = {}
for k in ENTITY_C2E_DICT:
    ENTITY_E2C_DICT[k] = get_exchange_dict(ENTITY_C2E_DICT[k])


PATH_DICT = {
    'xingchengdan.json':{
        'pic_label': '/root/maojingwei579/xingchengdan_annotated/label_from_picture/',
        'ocr_result': '/root/maojingwei579/意健险/ocr带坐标/xingchengdan/',
        'ocr_cal': '/root/maojingwei579/意健险/ocr带坐标_经纬矫正/xingchengdan/'
    },
    'boardcards.json':{
        'pic_label': '/root/maojingwei579/boarding_cards_annotated/label_from_picture/June08/',  # '/root/maojingwei579/意健险/业务方给定测试样本0803/登机牌/label.xlsx'
        'ocr_result': '/root/maojingwei579/意健险/ocr带坐标/boarding_cards/',
        'ocr_cal': '/root/maojingwei579/意健险/ocr带坐标_经纬矫正/boarding_cards/'
    },
    'xingchengdan_3code.json':{
        'pic_label': '/root/maojingwei579/xingchengdan_3code/label_from_picture/', # '/root/maojingwei579/意健险/业务方给定测试样本0803/行程单/label.xlsx'
        'ocr_result': '/root/maojingwei579/意健险/ocr带坐标/xingchengdan/',
        'ocr_cal': '/root/maojingwei579/意健险/ocr带坐标_经纬矫正/xingchengdan/'
    },
    'discharge_records.json':{
        'pic_label': '/root/maojingwei579/discharge_records_new/all_discharge_records_pic.xlsx'
    },
}

DEVICE_IDS = [0,1,2,3]

boardcards_model_dir = "20200503" # "boardcards/1500_288_augment"
xingchengdan_model_dir = "202003131520"
xingchengdan_3code_model_dir = "20200430"
discharge_records_model_dir = "discharge_records/keep_newline3" # "20200523"

CONFIG_DIC = {
        'boardcards':{
            'labels': './dataset/labels.txt',
            'config_name': './output/'+boardcards_model_dir+'/config.json',
            'model_name_or_path': '/root/maojingwei579/entity_recognize_for_flight/output/'+boardcards_model_dir+'/pytorch_model.bin',
            'tokenizer_name': './output/'+boardcards_model_dir+'/vocab.txt',
        },
        'xingchengdan':{
            'labels': './dataset/labels_xingchengdan.txt',
            'config_name': './output/'+xingchengdan_model_dir+'/config.json',
            'model_name_or_path': '/root/maojingwei579/entity_recognize_for_flight/output/'+xingchengdan_model_dir+'/pytorch_model.bin',
            'tokenizer_name': './output/'+xingchengdan_model_dir+'/vocab.txt',
        },
        'xingchengdan_3code':{
            'labels': './dataset/labels_xingchengdan_3code.txt',
            'config_name': './output/'+xingchengdan_3code_model_dir+'/config.json',
            'model_name_or_path': '/root/maojingwei579/entity_recognize_for_flight/output/'+xingchengdan_3code_model_dir+'/pytorch_model.bin',
            'tokenizer_name': './output/'+xingchengdan_3code_model_dir+'/vocab.txt',
        },
        "discharge_records":{
            "labels": "dataset/labels_discharge_records.txt",
            "config_name": "/root/maojingwei579/entity_recognize_for_flight/output/"+discharge_records_model_dir+"/config.json",
            "model_name_or_path": "/root/maojingwei579/entity_recognize_for_flight/output/"+discharge_records_model_dir+"/pytorch_model.bin",
            "tokenizer_name": "/root/maojingwei579/entity_recognize_for_flight/output/"+discharge_records_model_dir+"/vocab.txt"
        }}

OCR_TOKEN = os.environ.get("OCR_TOKEN", "67d93f36f27eff9e164d230d811b0f99") # "f06eab5c0a44e153e64039ff8a56cd1f" # prd
OCR_URL = os.environ.get("OCR_URL", "http://30.4.164.91:8060/ai/ocr/document_pos/v2")

PRD = os.environ.get("PRD", None)

PAT_DICT = {"出院记录":{}}
PAT_DICT["出院记录"]["sex"] = re.compile(r"性别[:：\s]*[男女]")
PAT_DICT["出院记录"]["age"] = re.compile(r"年龄[:：\s]*\d+")
PAT_DICT["出院记录"]["bed_number"] = re.compile(r"床[位号]号*[:：\s]*[A-Za-z\d\+\-\(\)一二区*]+")

# MRC_QUESTIONS = ['入院诊断是什么？', '出院情况是什么？', '出院诊断是什么？', '出院医嘱是什么？', '入院主诉是什么？']