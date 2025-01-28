import logging
import sys
import os
import os.path as osp


def init_logger(filename, logger_name):
    import datetime

    '''
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    '''

    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger
