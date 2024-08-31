import re
from rule_based.config_rule import *
from rule_based.flight_pattern import *


def flight_extraction(source_data, flight_pattern, replace):
    for pattern in flight_pattern:
        content = re.search(pattern, source_data.upper(), flags=re.IGNORECASE)
        if content:
            content = content.group().replace(' ', '').replace('\n', '')
            # print(content)
            if replace:
                for src, tgt in replace:
                    content = content.replace(src, tgt)
            break
    return content, source_data.replace(content, '  ') if content else source_data


if __name__ == '__main__':
    source_data = '张    0    国隆操保                  司资桃c医换锡          资格自隆换昌 Nanjing LuKou    ' \
                  'Internat ional Airpor t Gate登机口    20          Name姓名  XIONG/QIFENS目的地stination  ' \
                  '厦门XIAMEN        Name姓名      XI0NG/0IFENG姓名ame    X0NG/0IFEN 预计登机时Boarding Time    ' \
                  '1220      航班号    5C8812      日期                  航班号Flight No    SC8812          ' \
                  'Flight No.航班号SC8812 Seat No座位号    8D          船位Flight Nc                Date序号    ' \
                  '22N0V            Date日期        A              Date日期:    22N0V Class                  ' \
                  'SEQ No    02              座位号                    座位号 如登机口东标明镇起飞前2小时禹次确认 ' \
                  'Seaf Nc                            Seat Nc  07Po        ouns before      ETKD      ' \
                  '3242438522977/1 壹机口可能费更浦捌佰角广壹贰登机口提示信息    FOTV                          ' \
                  '        ' \
                  '  序号    021              序号    021 e yeto bo80as0r bo8rding gat6                    ' \
                  '                                            SEQ No.                            ' \
                  'SEQ No 重要提示:登机口路于起飞前15分钟关闭                                       ' \
                  '     ETKT                ETKT 费诊加0他参老费实                                 ' \
                  ' 441;1          0 元准 三附E雪 '

    flight_code_pattern = generate_pattern(file_path=code_path, key=code_key, pattern=code_pattern)
    flight_date_pattern = generate_pattern(file_path=date_path, key=date_key, pattern=date_pattern)

    res = flight_extraction(source_data, flight_code_pattern)
    print(res)