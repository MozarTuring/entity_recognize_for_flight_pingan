# ===========================================================================================
# flight code
# source_code_path = 'rule/flight_number.txt'
code_path = 'rule_based/flight_info.json'
code_key = 'total_flight_code'
code_pattern = [('', '\d{3,4}W?\s'),
                ('', '\s?\d{3,4}W?\s'),
                ('航班号?\s*', '\s?\d{3,4}W?'),
                ('flight?\s*', '\s?\d{3,4}W?'),
                ('', '\s?\d{3,4}W?')
                ]
flight_code_replace = [('航班', ''), ('号', ''), ('FLIGHT', '')]


# ===========================================================================================
# flight date
source_date_path = 'rule_based/flight_date.txt'
date_path = 'rule_based/flight_date.json'
date_key = 'flight_date'
date_pattern = [('[0123OIl][\dOIl]\s?\s?', '')
                ]
flight_date_replace = [('O', '0'), ('I', '1'), ('l', '1'), ('NAR', 'MAR'), ('NAY', 'MAY'), ('0CT', 'OCT'), ('N0V', 'NOV')]

# ===========================================================================================
# board_card path
# board_card_path = 'D:\需求\意健险单证\ocr结果\登机牌(已矫正)'
# board_card_path = 'D:\需求\意健险单证\ocr结果\登机牌第三批'

