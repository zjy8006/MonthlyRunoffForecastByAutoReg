#globalLog.py
import logging
import logging.config
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
if not os.path.exists(root_path+'/config/callrank_log/'):
    os.makedirs(root_path+'/config/callrank_log/')
  
def get_logger(name='root'):
    conf_log = os.path.abspath(os.getcwd() + "/config/logger_config.ini")
    logging.config.fileConfig(conf_log)
    return logging.getLogger(name)
 
 
logger = get_logger(__name__)
