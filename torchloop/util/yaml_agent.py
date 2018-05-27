# coding:utf-8
import yaml
import logging


def load_yaml(yaml_fname):
    logging.debug("yaml agent is loading file %s" % yaml_fname)
    with open(yaml_fname, 'r') as f:
        dic = yaml.load(f)
    return dic

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("testing yaml agent")
    dic = load_yaml("../amap/search_parameters/amap_request.yml")
    logging.debug("dic loaded %s" % str(dic))