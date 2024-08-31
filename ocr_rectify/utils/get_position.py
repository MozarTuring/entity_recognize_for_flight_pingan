from functools import reduce

WIN_SIZE = 3


def get_position(data):
    """
    从字典取4个点的坐标并返回
    :param data:
    :return:
    """
    pos_ls = data['coord']
    a = (pos_ls[0]['x'], pos_ls[0]['y'])
    b = (pos_ls[1]['x'], pos_ls[1]['y'])
    c = (pos_ls[2]['x'], pos_ls[2]['y'])
    d = (pos_ls[3]['x'], pos_ls[3]['y'])
    return a, b, c, d

def compute_font_size(font_ls):
    """
    计算字体大小
    :param font_ls:
    :return:
    """
    average_size = 0
    font_len = len(font_ls)
    for data in font_ls:
        a, b, c, d = get_position(data)
        average_size += (c[1]-b[1] + d[1]-a[1])/(2*font_len)
    return average_size

def get_ave_pos(data):
    """
    计算点A与D的均值，和B与C的均值，以及A、B、C、D四个点均值。
    :param data: dict not list
    :return:
    """
    a, b, c, d = get_position(data)
    return ((a[0]+d[0])/2, (a[1]+d[1])/2), ((b[0]+c[0])/2, (b[1]+c[1])/2), ((a[0]+d[0]+b[0]+c[0])/4, (a[1]+d[1]+b[1]+c[1])/4)

def get_min_y(data):
    """
    by jingwei mao
    """
    a, b, c, d = get_position(data)
    return min(a[1], b[1], c[1], d[1])

def position_sort(data_tuple):
    """
    先寻找同一行，将同一行的内容放在一下list，之后再对这个同行的list内容进行先后顺序的调整。
    :param data_tuple:
    :return:
    """
    #

    pass

def row_combine(info):
    """
    将本应该属于同一行的单元存储到同一个list
    :param info:
    :return:
    """
    out = [[]]
    out[0] = [info[0]]
    _, _, (_, y_pre_ave) = get_ave_pos(info[0])

    for idx in range(1, len(info)-1):
        font_size = compute_font_size(info[idx - 1:idx + 2])
        _, _, (_, y_cur_ave) = get_ave_pos(info[idx])
        # print(y_pre_ave, y_cur_ave, y_cur_ave-y_pre_ave, font_size)
        if y_cur_ave >= y_pre_ave + font_size:
            out.append([info[idx]])
        else:
            out[-1].append(info[idx])
        y_pre_ave = y_cur_ave
        
    _, _, (_, y_cur_ave) = get_ave_pos(info[-1])
    if y_cur_ave >= y_pre_ave + font_size:
        out.append([info[-1]])
    else:
        out[-1].append(info[-1])
    return out


def row_combine_new(info):
    """
    by jingwei mao
    """
    out = [[]]
    out[0] = [info[0]]
    _, _, (_, y_pre_ave) = get_ave_pos(info[0])

    for idx in range(1, len(info)-1):
        _, _, (_, y_cur_ave) = get_ave_pos(info[idx])
        if get_min_y(info[idx]) >= y_pre_ave:
            out.append([info[idx]])
        else:
            out[-1].append(info[idx])
        y_pre_ave = y_cur_ave
        
    # _, _, (_, y_cur_ave) = get_ave_pos(info[-1])
    if get_min_y(info[-1]) >= y_pre_ave:
        out.append([info[-1]])
    else:
        out[-1].append(info[-1])
    return out


def concat_unit(line, pos):
    """
    对处于同一行的元素进行拼接。
    :param line: 排过序后的list [{},{}]
    :param pos: 排序后的[(idx, x_start, x_end, x_center), (), ()...]
    :return:
    """
    font_size = compute_font_size(line)
    pre_x_start = pos[0][1]
    pre_x_end = pos[0][2]
    for idx in range(1, len(pos)):
        cur_x_start = pos[idx][1]
        cur_x_end = pos[idx][2]
        space =  '  '*int((cur_x_start - pre_x_end)/font_size + 2)
        line[idx-1]['text'] = line[idx-1]['text']+ space
        pre_x_end = cur_x_end
    return line

def order_rectify(out):
    """
    核心汇总算法，集成了行矫正，行中列的矫正，以及分段矫正算法
    :param out:
    :return:
    """
    ordered_out = []
    for line in out:
        if len(line) == 1:
            ordered_out.append(line)
        else:
            pos = [get_ave_pos(t) for t in line]
            pos = [(idx, x_start, x_end, x_center) for idx, ((x_start, _), (x_end, _), (x_center, _)) in enumerate(pos)]
            # 根据x_center进行排序，来进行行内次序矫正。
            pos.sort(key=lambda x: x[-1])
            ordered = [line[i[0]] for i in pos]
            # 根据调整后的pos进行行内元素的拼接
            ordered = concat_unit(ordered, pos)
            ordered_out.append(ordered)
    return ordered_out

def split_section(ordered_out, info):
    line_end = [x_ave for ((_, _), (x_ave, _), (_, _)) in [get_ave_pos(data) for data in info]]
    x_max = max(line_end)
    out = []
    # out[0] = ordered_out[0]
    # (_, _), (pre_x_end, _), (_, _) = get_ave_pos(out[0][-1])
    for line_ls in ordered_out:
        (_, _), (cur_x_end, _), (_, _) = get_ave_pos(line_ls[-1])
        font_size = compute_font_size(line_ls)
        out.append(line_ls)
        if x_max - cur_x_end >= 2*font_size:
            # if line_ls[-1]['text'][-1] ==
            #     continue
            # else:
                out.append(['\n'])
    return out


def split_section_simple(ordered_out, info):
    line_end = [x_ave for ((_, _), (x_ave, _), (_, _)) in [get_ave_pos(data) for data in info]]
    x_max = max(line_end)
    out = []
    for line_ls in ordered_out:
        # (_, _), (cur_x_end, _), (_, _) = get_ave_pos(line_ls[-1])
        # font_size = compute_font_size(line_ls)
        out.append(line_ls)
        out.append(['\n'])
    return out


def split_section_new(ordered_out, info):
    line_end = [x_ave for ((_, _), (x_ave, _), (_, _)) in [get_ave_pos(data) for data in info]]
    x_max = max(line_end)
    to_be_ordered = []
    new_ordered_out = []
    count = 0
    x_end_ls = []
    for line_ls in ordered_out:
        (_, _), (cur_x_end, _), (_, _) = get_ave_pos(line_ls[-1])
        font_size = compute_font_size(line_ls)
        x_end_ls.append((count,cur_x_end))
        to_be_ordered.append(line_ls)
        count += 1
        if x_max - cur_x_end >= 2*font_size:
            # import ipdb; ipdb.set_trace()
            x_end_ls.sort(key=lambda x: x[-1])
            for (x, _) in x_end_ls:
                new_ordered_out.append(to_be_ordered[x])

            new_ordered_out.append(['\n'])
            x_end_ls = []
            to_be_ordered = []
            count = 0
    return new_ordered_out


def generate_content(out):
    result = ''
    for line in out:
        if len(line) == 1:
            if line == ['\n']:
                result = result + '\n'
            else:
                # print(line)
                result = result + line[0]['text']
        else:
            result = result + reduce(lambda x, y: x + y, [t['text'] for t in line])
    return result
