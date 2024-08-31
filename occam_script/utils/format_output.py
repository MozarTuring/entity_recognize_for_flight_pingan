for ind_ls, pred_ls in enumerate(pred_lls):
    entity_ls = []
    entity = ''
    flag = False
    for ind_pred, pred in enumerate(pred_ls):
        if pred == 2:
            if flag:
                entity = entity + tokens_fs[ind_batch * config.batch_size + ind_ls][ind_pred]
            else:
                entity_ls.append(entity)
                entity = tokens_fs[ind_batch * config.batch_size + ind_ls][ind_pred]
                flag = True
        elif pred == 3:
            if flag:
                entity = entity + tokens_fs[ind_batch * config.batch_size + ind_ls][ind_pred]
        else:
            flag = False
