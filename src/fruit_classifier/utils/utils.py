import os
import yaml
import sys

from fruit_classifier.logger.logger import logging
from fruit_classifier.exception.exception import CustomException


def read_yaml(path_to_yaml):

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        logging.info('Error occured while reading the yaml file.')
        raise CustomException(e, sys)
    

def create_directories(path_to_directories):

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")



