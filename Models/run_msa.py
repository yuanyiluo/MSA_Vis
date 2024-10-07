from deepface import DeepFace
from PyQt5.QtCore import QThread, pyqtSignal
from .msa.models import *
import torch
from pathlib import Path
import json
from easydict import EasyDict as edict
from collections import OrderedDict
from .data_process.get_Vfeatures import GetFeatures
from .data_process.get_Vfeatures import GetFeatures


class MSA:
    # msa_signal = pyqtSignal(str)

    def __init__(self):
        super(MSA, self).__init__()
        self.Model_MAP = {
            'lf_dnn': LF_DNN,
        }

    def msa(self, sync, model: str, video_name: str, data):

        # get data
        t_feature = data[0]
        a_feature = data[1]
        v_feature = data[2]

        if v_feature.any() and a_feature.any() and t_feature.any():
            args = self.get_config_regression(model)
            run_model = self.Model_MAP[model](args)

            with torch.no_grad():
                state_dict = torch.load(f'D:/Research/code/MSA_Vis/Models/msa/model_weights/{model.lower()}-sims.pth')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k[:6] == 'Model.':
                        k = k[6:]  # remove `module.`
                    new_state_dict[k] = v

                run_model.load_state_dict(new_state_dict)
                run_model.eval()
                msa_res = run_model(t_feature, a_feature,
                                    v_feature)  # dict{'Feature_t', 'Feature_a', 'Feature_v', 'Feature_f', 'M'}
                return msa_res

        else:
            print("there are no features")

    def get_config_regression(self, model_name, config_file=""):
        if config_file == "":
            config_file = Path(__file__).parent / "msa" / "model_config" / "config_regression.json"
        with open(config_file, 'r') as f:
            config_all = json.load(f)
        model_common_args = config_all[model_name]['commonParams']
        model_dataset_args = config_all[model_name]['datasetParams']['sims']
        dataset_args = config_all['datasetCommonParams']['sims']
        # use aligned feature if the model requires it, otherwise use unaligned feature
        if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        else:
            dataset_args = dataset_args['unaligned']

        config = {}
        config.update(dataset_args)
        config.update(model_common_args)
        config.update(model_dataset_args)
        config = edict(config)  # use edict for backward compatibility with MMSA v1.0
        return config

    def msa2(self, img):
        r = DeepFace.analyze(img, actions=['emotion'], detector_backend='mtcnn')
        return r


if __name__ == '__main__':
    msa = MSA()
    res = msa.msa({'text': 'hello', 'audio': 'audio', 'video': 'video'}, 'lf_dnn')
    print(res)
