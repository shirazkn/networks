import os, sys, json


def clearfile(path):
    if os.path.isfile(path):
        os.remove(path)
    return

def check_extension(filename, ext):
    if not filename.split(".")[-1] == ext:
        filename += "." + ext
    return filename

def exit():
    os._exit(0)

def sys_exit():
    sys.exit()

def dump(data, filename):
    filename = check_extension(filename, 'json')
    clearfile(filename)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load(filename):
    filename = check_extension(filename, 'json')
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data