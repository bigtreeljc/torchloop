import logging
import os

# define logging level of the fence project
DEBUG=logging.DEBUG
INFO=logging.INFO
WARNING=logging.WARNING
ERROR=logging.ERROR
CRITICAL=logging.CRITICAL

logger_ = logging.getLogger()
logger_.setLevel(INFO)
ch = logging.StreamHandler()
ch.setLevel(DEBUG)
formatter = logging.Formatter('%(asctime)s-%(name)s-' + \
        '%(levelname)s-%(filename)s:%(lineno)d-%(message)s')
ch.setFormatter(formatter)
logger_.addHandler(ch)

# also allow user to change the debug_level
def tl_logger(log_level=CRITICAL, update_level=False):
    if update_level:
        logging.getLogger().setLevel(log_level)
        # ch.setLevel(log_level)
    return logging.getLogger()

def setup_file_handler(log_fname, log_level=CRITICAL):
    fh = logging.FileHandler(log_fname)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger_.addHandler(fh)

def setlogger(debug_level):
    switcher = {
        0: CRITICAL,
        1: ERROR,
        2: WARNING,
        3: INFO,
        4: DEBUG
    }

    debug_levels = {
        10: "10-debug",
        20: "20-info",
        30: "30-warning",
        40: "40-error",
        50: "50-critical"
    }

    d_level = debug_level
    try:
        d_level = switcher[d_level]
    except Exception as e:
        print(e.args)

    logging.getLogger().info("switching debug level {}".\
            format(d_level))
    tl_logger(log_level=d_level, update_level=True)
