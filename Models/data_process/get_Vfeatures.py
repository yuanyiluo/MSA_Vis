from deepface import DeepFace
import os
import shutil
import subprocess
import pandas as pd


class GetFeatures:
    def __init__(self, name):
        self.openface2Path = "D:/Research/tools/OpenFace_2.2.0_win_x64/FeatureExtraction.exe"
        self.name = name
        self.text = ''

    def getVideoEmbedding(self):
        print('start get video embedding')
        cmd = f"{self.openface2Path} -f {self.name}"
        try:
            subprocess.run(cmd)
            print("标准输出: 完成特征提取")  # 打印标准输出
        except subprocess.CalledProcessError as e:
            print(f"运行出错: {e}")

        # 删除特征文件之外的文件
        self.delete_except("D:/Research/code/MSA_Vis/Control/processed")

        v_embedding = self.__get_Vembedding().mean(0).reshape(1, -1)
        return v_embedding

    def delete_except(self, folder_path):  # 删除特征文件之外的文件
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 构建完整路径
            file_path = os.path.join(folder_path, filename)
            # 检查是否是文件，并且不是例外文件
            if os.path.isfile(file_path) and not filename.endswith('.csv'):
                os.remove(file_path)  # 删除文件
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # 读取视频特征文件
    def __get_Vembedding(self):
        path = f"D:/Research/code/MSA_Vis/Control/processed/{self.name.split('/')[-1][:-3]}csv"
        df = pd.read_csv(open(path, 'r'))
        if len(df) > 40:
            v_embedding = df.iloc[1:40, 5:].to_numpy()
        else:
            v_embedding = df.iloc[1:, 5:].to_numpy()
        return v_embedding

    # def getSentiment(self, imgs):
    #     # v_embedding_list = []
    #     for img in imgs:
    #         emb = DeepFace.represent(
    #             img,
    #             model_name='OpenFace',
    #         )
    #         v_embedding_list.append(emb)
    #         print(len(emb[0]['embedding']))
    #
    #     1 / 0


if __name__ == '__main__':
    getFeatures = GetFeatures('test')
    getFeatures.getTextEmbedding()
