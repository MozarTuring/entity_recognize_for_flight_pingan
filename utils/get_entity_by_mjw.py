symbol_tofilter_set = ['。', '【', '】']
symbol_toremove_set = set(['\u200b', '“', '”', '\xa0', '\u3000'])

train_entities = []
with open('/data/maojingwei/finicial_entity/train_entities.txt', 'r') as rf_train_ent:
    for line in rf_train_ent:
        train_entities.append(line.strip('\n'))

def filter_word(input_str):
    output_str = input_str
    for sbtof in symbol_tofilter_set:
        if sbtof in output_str:
            output_str = output_str.replace(sbtof, '')

    return output_str


def create_new_ls(temp_str_ls):
    new_temp_str_ls = []
    for temp_str1 in set(temp_str_ls):
        temp_str1 = filter_word(temp_str1)
        flag = False
        for temp_str2 in set(temp_str_ls):
            temp_str2 = filter_word(temp_str2)
            if temp_str1.lower() == temp_str2.lower() and temp_str1 != temp_str2:
                flag = True
                break

        if not flag:
            new_temp_str_ls.append(temp_str1)
        else:
            count1, count2 = 0, 0
            for str_ele1, str_ele2 in zip(temp_str1, temp_str2):
                if str_ele1.isupper():
                    count1 += 1
                if str_ele2.isupper():
                    count2 += 1
            if count1 >= count2 and temp_str1 not in new_temp_str_ls:
                new_temp_str_ls.append(temp_str1)
            if count1 < count2 and temp_str2 not in new_temp_str_ls:
                new_temp_str_ls.append(temp_str2)

    # 去除短的重复实体（注意要全部小写或大写再比较）
    new_temp_str_ls2 = []
    for new_temp_str1 in new_temp_str_ls:
        flag = False
        for new_temp_str2 in new_temp_str_ls:
            if new_temp_str1.lower() in new_temp_str2.lower() and new_temp_str1 != new_temp_str2:
                flag = True
                break
        if not flag:
            new_temp_str_ls2.append(new_temp_str1)

    # 去除长度为1的和含有\u200b(vim打开显示为<200b>)等奇怪字符的实体
    new_temp_str_ls3 = []
    for new_temp_str in new_temp_str_ls2:
        if len(new_temp_str) != 1 and not set(new_temp_str)&symbol_toremove_set and new_temp_str not in train_entities:
            new_temp_str_ls3.append(new_temp_str)
    
    return new_temp_str_ls3


def extract_entity(input_pred_lls, input_tokens_fs, input_ind_batch, batch_size, text_ids=None, wf_csv=None):
    # input_pred_lls: 预测的label列表，double list
    # input_token_fs: 文本列表，double list
    # input_ind_batch: batch编号
    new_entity_lls = []
    for ind_ls, pred_ls in enumerate(input_pred_lls):
        entity_ls = []
        entity = ''
        flag = False
        for ind_pred, pred in enumerate(pred_ls):
            if pred == 2: # 实体开始编号，根据实际修改
                if flag:
                    entity = entity + input_tokens_fs[input_ind_batch*batch_size+ind_ls][ind_pred]
                else:
                    entity_ls.append(entity)
                    entity = input_tokens_fs[input_ind_batch*batch_size+ind_ls][ind_pred]
                    flag = True
            elif pred == 3: # 实体中间编号，根据实际修改
                if flag:
                    entity = entity + input_tokens_fs[input_ind_batch*batch_size+ind_ls][ind_pred]
            else:
                flag = False
            
        new_entity_ls = create_new_ls(entity_ls)
        if wf_csv and text_ids:    
            wf_csv.writerow([text_ids[input_ind_batch*batch_size+ind_ls], ';'.join(new_entity_ls)])
        
        new_entity_lls.append(new_entity_ls)
    
    return new_entity_lls