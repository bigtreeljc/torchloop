from torchloop.util import config_object as co

class Ipipeline(co.configurable):
    def __init__(self, yaml_or_dic, **additional_conf):
        super(Ipipeline, self).__init__(yaml_or_dic, **additional_conf)
        self.setup_helper()

    def setup_helper(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class Irunner(co.configurable):
    def __init__(self, yaml_or_dic, **additional_conf):
        super(Irunner, self).__init__(yaml_or_dic, **additional_conf)
        self.setup_helper()

    def setup_helper(self):
        logger.debug("does nothing in setup helper")

    def run(self):
        raise NotImplementedError
