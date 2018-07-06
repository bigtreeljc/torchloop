from torchloop.util.arg_parse_object as apo
from torchloop.util import reflection, yaml_agent, tl_logging

class run:
    def __init__(self):
        self.parser = apo.generic_argparser()

    def __call__(self, *args, **kwargs):
        parser = self.parser.get_parser()
        debug_level = parser.debug_level
        conf_file = parser.conf_file
        try:
            self.conf = yaml_agent.load_yaml(conf_file)
            # utils.setlogger(debug_level)
            tl_logging.setlogger(debug_level)
            logger.info("parser is {}".format(parser))
        except FileNotFoundError:
            print("conf file {} not found, exiting".format(conf_file))
        except Exception as e:
            raise e

        pipeline_clsname: str = None
        # setup deploy_helper arguments to arg parsers
        runner_helper_dic = self.conf["args"]
        try:
            parser = self.parser.deploy_helper_args(
                    runner_helper_dic)[0]
        except Exception as e:
            pass
        self.reconfig_deploy_helper(parser, runner_helper_dic)
        logger.info("after update deploy_helper_dic {}".format(
            self.conf["args"]))

        try:
            runner_clsname = self.conf["run_cls"]
            logger.debug("run class {}".format(runner_clsname))
            runner_cls = reflection.for_name(runner_clsname)
            self.runner = runner_cls(self.conf)
            logger.info("runner cls {}".format(
                self.runner.__class__.__name__))
        except Exception as e:
            logger.error("error try to initialize pipeline {}".format(
                pipeline_clsname))
            raise e

        self.runner.run()

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

def main():
    run_ = run()
    run_()

if __name__ == "__main__":
    main()
