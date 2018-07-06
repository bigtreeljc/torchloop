import time
from util import tl_logging
logger = tl_logging.tl_logger()

def time_d(method):
    def timed(*args, **kw):
        ts = time.time()
        logger.debug("begin to time phase for method {}".format(
            method.__name__))
        result = method(*args, **kw)
        te = time.time()
        logger.debug('method %r time %2.2f ms' % (method.__name__, 
            (te - ts) * 1000))
        return result
    return timed
