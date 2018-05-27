import json

def load_json_file(fname):
    '''
    :param fname:fname endswith json
    :return:
    '''
    with open(fname, "r", encoding='UTF-8') as f:
        dic = json.load(f)
    return dic

def dump_to_json_file(fname, json_seriasable):
    with open(fname, 'w', encoding='UTF-8') as f:
        json.dump(json_seriasable, f, ensure_ascii=False, indent=2)
