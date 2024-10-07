from Client.listener.detector import Detector
import numpy as np
import pandas as pd
import librosa
from time import sleep
from transformers import BertModel, BertTokenizer
import torch
from PyQt5.QtCore import QThread, pyqtSignal
import config


class GetATFeatures():
    # feature_signal = pyqtSignal()
    def __init__(self):
        super(GetATFeatures, self).__init__()
        self.features = []
        self.ad = Detector()
        self.ad.audio_detector()

    def getAudioEmbedding(self):  # , output_queue, sync_event
        # while True:
        try:
            data_queue = self.ad.data_queue
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():  # 当听不到声音后，开始transform
                print('start get audio embedding')
                text_data, audio_data = data_queue.get()
                print("text is:", text_data)
                data_queue.queue.clear()

                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data[:-1]  # 截断最后一个字节
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # using librosa to get audio features (f0, mfcc, cqt)
                hop_length = 512  # hop_length smaller, seq_len larger
                f0 = librosa.feature.zero_crossing_rate(audio_np, hop_length=hop_length).T  # (seq_len, 1)
                mfcc = librosa.feature.mfcc(y=audio_np, sr=16000, hop_length=hop_length,
                                            htk=True).T  # (seq_len, 20)
                cqt = librosa.feature.chroma_cqt(y=audio_np, sr=16000, hop_length=hop_length).T  # (seq_len, 12)
                tem_feature = np.concatenate([f0, mfcc, cqt], axis=-1)
                self.features.append(tem_feature)
                audio_feature = np.concatenate(self.features).mean(0).reshape(1, -1)
                self.features = []

                # get text feature
                text_feature = self.getTextEmbedding(text_data)

                # Infinite loops are bad for processors, must sleep.
                return (audio_feature, text_feature, text_data)
                # print(audio_feature.shape, text_feature.shape)
                # output_queue.put(text_feature, audio_feature)
                # sync_event.set()
                # sleep(0.25)
        except KeyboardInterrupt:
            print('error')
            # break

    def getTextEmbedding(self, text_data):
        print('start get text embedding')
        if text_data != '':
            # 加载预训练模型和分词器
            bert_mode = "D:/Research/code/MSA_Vis/Models/bert_cn"
            tokenizer = BertTokenizer.from_pretrained(bert_mode)
            model = BertModel.from_pretrained(bert_mode)

            model.eval()  # 将模型设置为评估模式
            inputs = tokenizer(text_data, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
            with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
            text_features = last_hidden_states[:, 0, :].numpy()  # 提取 [CLS] 标记的特征last_hidden_states[:, 0, :]
            return text_features
