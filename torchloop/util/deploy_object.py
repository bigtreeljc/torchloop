from torchloop.util import reflection, /
        arg_parse_object, yaml_agent
import sys
from torchloop.util import torchloop_logging
logger = torchloop_logging.tl_logger()

class deploy:
    def __init__(self):
        self.parser = arg_parse_object.generic_argparser()

    def __call__(self, *args, **kwargs):
        parser = self.parser.get_parser()#.parse()
        debug_level = parser.debug_level
        conf_file = parser.conf_file
        try:
            self.conf = yaml_agent.load_yaml(conf_file)
            torchloop_logging.setlogger(debug_level)
            logger.info("parser is {}".format(parser))
        except FileNotFoundError:
            print("conf file {} not found, exiting".format(conf_file))
        except Exception as e:
            raise e

        pipeline_clsname = None
        # setup deploy_helper arguments to arg parsers
        deploy_helper_dic = self.conf["run_helper"]
        try:
            parser = self.parser.deploy_helper_args(deploy_helper_dic)[0]
        except Exception as e:
            pass
        self.reconfig_deploy_helper(parser, deploy_helper_dic)
        logger.info("after update deploy_helper_dic is {}".format(
            self.conf["deploy_helper"]))

        try:
            pipeline_clsname = self.conf["pipeline_cls"]
            logger.debug("pipeline name is {}".format(pipeline_clsname))
            pipeline_cls = reflection.for_name(pipeline_clsname)
            self.pipeline = pipeline_cls(self.conf)
            logger.info("pipeline cls is {}".format(
                self.pipeline.__class__.__name__))

        except Exception as e:
            logger.error("error try to initialize pipeline {}".format(
                pipeline_clsname))
            raise e

        self.pipeline.run()

    def reconfig_deploy_helper(self, parser, deploy_helper_dic):
        substitude_lv = {}
        logger.debug("parser is {}".format(parser))
        logger.debug("before reconfig {}".format(deploy_helper_dic))
        for k, v in deploy_helper_dic.items():
            if hasattr(parser, k):
                logger.debug("k, v substitude {}, {}".format(
                    k, v
                ))
                v_substitude = getattr(parser, k)
                substitude_lv[k] = v_substitude
        deploy_helper_dic.update(substitude_lv)

