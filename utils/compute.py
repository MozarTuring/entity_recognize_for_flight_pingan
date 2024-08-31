def compute_F1(data, label_field, label_target_field, predict_filed, predict_target_field):
    """
    计算某个字段的正确率、召回率、F1，data之一元素如下。一次只能算一个字段。
    {'Label': {'Loc': ['沿江村7组82号'], 'NUM': ['3000'], 'PER': ['刘双彩', '魏红叶', '顾光兵'], 'TIME': ['2019年8月15日11点']},
     'Origin': '被保人为....',
     'Predict': {'LOC': ['沿江村7组'], 'NUM': ['11', '3000'], 'ORG': ['沭阳县中医院住院治疗中眼科'], 'PER': ['魏红叶', '刘双彩'],
                'TIME': ['2019年8月15日', '25号'],
                'TITLE': ['顾光兵']}}
    :param data: list, 某个元素是dict: {'Label':{}, 'Predict':{}, ...}
    :param label_field: 如上的'Label'等第一级key
    :param label_target_field: 如上的'Loc'等第二级key.
    :param predict_filed: 同上
    :param predict_target_field: 同上
    :return: F1, accuracy, recall
    """
    n_correct_recognized_entity = 0
    n_recognized_entity = 0
    n_total_entity = 0
    def compute_F1_factor(label, predict):
        n_total_entity = len(label)
        n_correct_recognized_entity = sum([1 if t in label else 0 for t in predict])
        n_recognized_entity = len(predict)
        return n_total_entity, n_recognized_entity, n_correct_recognized_entity
    for i in data:
        if label_target_field in i[label_field]:
            label = i[label_field][label_target_field]
        else:
            label = []
        if predict_target_field in i[predict_filed]:
            predict = i[predict_filed][predict_target_field]
        else:
            predict = []
        F1_factors = compute_F1_factor(label, predict)
        n_total_entity += F1_factors[0]
        n_recognized_entity += F1_factors[1]
        n_correct_recognized_entity += F1_factors[2]
    accuracy = n_correct_recognized_entity/n_recognized_entity
    recall = n_correct_recognized_entity/n_total_entity
    F1 = (2*accuracy*recall)/(accuracy+recall)
    return F1, accuracy, recall