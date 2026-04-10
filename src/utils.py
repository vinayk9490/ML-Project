import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    '''
    The pickel file contains the preprocessor that tells what column is numerical 
    and what column is categorical and how to transform them. 
    This function is responsible for saving the preprocessor object as a pickle file.
    '''
    logging.info("Entered the save_object method of utils")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
