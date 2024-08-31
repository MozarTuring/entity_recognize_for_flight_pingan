import os
import sys
import json
from utils.get_position import *

# WIN_SIZE = 3

image_dir = r'D:\理赔单证\理赔单证ocr加坐标\出院记录'

def format_rectify(image_dir):
    for d, _, p in os.walk(image_dir):
        if p:
            for fp in p:
                if '.json' in fp:
                    out = [[]]
                    path = os.path.join(image_dir, d, fp)
                    with open(path) as f:
                        data = json.loads(f.read())
                        # print(data)
                    if data['result']['message'] == 'success' and data['error'] == '':
                        info = data['result']['info']
                        out = row_combine(info)
                        out = order_rectify(out)
                        out = split_section(out, info)
                        out = generate_content(out)
                        with open(path.replace('.json', '_new.txt'), mode='w') as f:
                            f.write(out)
                        print(out)

if __name__ == '__main__':
   image_path  = sys.argv[-1]
   format_rectify(image_path)