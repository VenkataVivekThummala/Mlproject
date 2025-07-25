import os
import sys
import dill
import pandas as pd
import numpy as np

from src.exception_handling import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as ex:
        raise CustomException(ex,sys)

        