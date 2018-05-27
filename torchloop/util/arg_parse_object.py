import argparse
import sys

class iov_argparser_generic:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def get_parser(self):
        raise NotImplementedError

class generic_argparser(iov_argparser_generic):
    def get_parser(self):
        self.parser.add_argument("--conf_file",
                            help="yaml conf_file that you want to run",
                            dest="conf_file",
                            required=True,
                            type=str)
        self.parser.add_argument("--debug_level",
                                 dest="debug_level",
                                 type=int,
                                 default=1)

        return self.parser.parse_known_args()[0]

    def _dynamic_conf_parser(self, kv_pairs):
        for key__, value__ in kv_pairs.items():
            self.parser.add_argument("--{}".format(key__),
                                     dest=key__,
                                     type=type(value__),
                                     default=value__
                                    )
    
    def deploy_helper_args(self, deploy_helper_dic):
        self._dynamic_conf_parser(deploy_helper_dic)
        args = sys.argv[1:]
        return self.parser.parse_known_args(args)

class rewrite_garloc_argparser(iov_argparser_generic):
    def get_parser(self):
        self.parser.add_argument("--conf_file",
                            help="yaml conf_file that you want to run",
                            dest="conf_file",
                            type=str)
        self.parser.add_argument("--debug_level",
                                 dest="debug_level",
                                 type=int)

        return self.parser.parse_args
