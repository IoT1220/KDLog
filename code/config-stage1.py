from enum import Enum
from prompts import *


class DatasetEnum(Enum):
    cmcc: str = "cmcc"
    zte: str = 'zte'
    aliyun: str = 'aliyun'


def get_system_prompt(dataset: DatasetEnum) -> str:
    return {
        DatasetEnum.cmcc: cmcc_system_prompt,
        DatasetEnum.zte: zte_system_prompt,
        DatasetEnum.aliyun: aliyun_system_prompt,
    }[dataset]

def get_fewshot_examples_path(dataset: DatasetEnum) -> str:
    return {
        DatasetEnum.cmcc: 'data_partitioning_CMCC0805/train3+test3/15shotcorrect.json',
        DatasetEnum.zte: 'zte_newdata1028/fewshot27.json',
        DatasetEnum.aliyun: '/home/xuting/2.orpo/data/Aliyun_labeled_1217/train/fewshot9-correct.json'
    }[dataset]

def get_data_folder(dataset: DatasetEnum) -> str:
    return {
        DatasetEnum.cmcc: 'data_partitioning_CMCC0805',
        DatasetEnum.zte: '/home/xuting/2.orpo/01_zte/train4_0.8_labeled48-unlabeled12',
        DatasetEnum.aliyun: '/home/xuting/2.orpo/00_aliyun/train4_0.8_labeled48-unlabeled12',
    }[dataset]
