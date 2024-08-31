import os, sys, json


wf = open("tmp.txt", 'w', encoding='utf8')
base_dir = sys.argv[-1]
def check_ocr_error(inp_dir):
    for root, dirs, files in os.walk(inp_dir):
        if files:
            if not os.path.exists(os.path.join(root, "log.json")):
                wf.write(os.path.join(root, "log.json") + 'not exist \n')
                return None
            with open(os.path.join(root, "log.json"), 'r', encoding='utf8') as rf:
                loaded = json.load(rf)
                print(loaded[0]["code"])
                if loaded[0]["code"] in ["10002", "10006"]:
                    wf.write(inp_dir+'\n')
                    return None

if __name__ == "__main__":
    for dir in os.listdir(base_dir):
        check_ocr_error(os.path.join(base_dir, dir))
    wf.close()
