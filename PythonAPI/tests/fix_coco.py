import pytest 
from loguru import logger
import json

file = '/opt/product/cocoapi/PythonAPI/tests/predictions.json'
# read file 
with open(file, 'r') as f:
    data = json.load(f)
    logger.info(f'load data from {file}')
    logger.info(f'data type {type(data)}')
    logger.info(f'data len {len(data)}')
for d in data:
    d['image_id'] = int(d['image_id'])
json.dump(data, open(file, 'w'))