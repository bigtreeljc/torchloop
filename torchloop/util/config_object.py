from torchloop.util import yaml_agent
from pydoc import locate

class dic_first_layer_config:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

class config(dic_first_layer_config):
    def __init__(self, dic):
        super(config, self).__init__(dic)

class configurable:
    def __init__(self, yaml_or_dic_conf=None, **additional_conf):
        if isinstance(yaml_or_dic_conf, str):
            conf_dic = yaml_agent.load_yaml(yaml_or_dic_conf)
        elif isinstance(yaml_or_dic_conf, dict):
            conf_dic = yaml_or_dic_conf
        elif yaml_or_dic_conf is None:
            conf_dic = {}
        else:
            raise AttributeError("yaml_or_dic_conf is not fname or dict")
        self.conf = conf_dic
        self._init_from(conf_dic)
        self._init_from(additional_conf)
        self.extra_conf()
        self.init_all_cls()

    def extra_conf(self):
        pass

    def _init_from(self, dic):
        try:
            self.conf_keys()
        except NotImplementedError as e:
            pass

        has_conf = hasattr(self, 'conf_keys_')
        for k, v in dic.items():
            if not has_conf or (has_conf and k in self.conf_keys_):
                # print("class %s attr %s" % (__class__, k))
                setattr(self, k, v)

    def _init_from_dic(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

    def config(self, dic):
        self._init_from_dic(dic)

    def config_with_variables(self, **dic):
        self._init_from_dic(dic)

    def conf_keys(self):
        raise NotImplementedError

    def validate_myself(self, keys=[]):
        if isinstance(keys, str):
            assert hasattr(self, keys)
            if hasattr(self, "conf_keys_"):
                assert keys in self.conf_keys_
        elif isinstance(keys, list):
            validate_keys = keys if len(keys) > 0 else self.conf_keys()
            for k in validate_keys:
                if not hasattr(self, k):
                    raise AttributeError("myself is not validated key %s not in" % k)
                if hasattr(self, "conf_keys_"):
                    assert k in self.conf_keys_
        else:
            raise AttributeError("can't validate keys with invalidate type %s" % type(keys))

    def _is_cls_key(self, key_):
        return key_.endswith('cls'):

    def _dfs_traverse(self, cur_conf, cls_cb=self.init_cls_cb):
        for k_ in cur_conf.keys():
            if isinstance(cur_conf[k_], dict):
                self._dfs_traverse(cur_conf[k_], cls_cb)
            else:
                if self._is_cls_key(k_):
                    cur_conf[k_] = cls_cb(cur_conf[k_])

    def init_cls_cb(self, v_):
        cls = locate(v_)
        return cls

    def init_all_cls(self, cur_conf, kv_cb):
        self._dfs_traverse(self.conf)
