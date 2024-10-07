import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PyQt5.QtCore import QThread
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.backends.backend_agg as agg


class Thread_Feature_Visualization():  # QThread
    # features_signal = QtCore.pyqtSignal()

    def __init__(self, data):
        super(Thread_Feature_Visualization, self).__init__()
        # self.t_feature = data['feature_t']
        # self.a_feature = data['feature_a']
        # self.v_feature = data['feature_v']
        # self.m_feature = data['feature_m']
        self.m_feature = data

    def run(self):  # , data
        # self.t_feature.append(data['Feature_t'])
        # self.a_feature.append(data['Feature_a'])
        # self.v_feature.append(data['Feature_v'])
        # self.m_feature.append(data['Feature_m'])
        # t_feature = np.squeeze(np.stack(self.t_feature), axis=1)
        # a_feature = np.squeeze(np.stack(self.a_feature), axis=1)
        # v_feature = np.squeeze(np.stack(self.v_feature), axis=1)
        m_feature = np.squeeze(np.stack(self.m_feature), axis=1)

        # 应用 PCA 从 128 维降至 2 维
        pca = PCA(n_components=2)
        # t_reduced = pca.fit_transform(t_feature)
        # a_reduced = pca.fit_transform(a_feature)
        # v_reduced = pca.fit_transform(v_feature)
        m_reduced = pca.fit_transform(m_feature)

        # features = [t_reduced, a_reduced, v_reduced, m_reduced]

        # 准备不同的颜色和标签
        color = 'red'  # , 'blue', 'green', 'purple'
        labels = 'Feature m'  # 'Feature t', 'Feature a', 'Feature v',

        # 可视化
        fig = plt.figure(figsize=(8, 6))
        # for i, color in enumerate(colors):
        plt.scatter(m_reduced[:, 0], m_reduced[:, 1], color=color, label=labels, s=150, edgecolors='k')
        plt.title('Feature Visualization')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # plt.show()

        # 将 Matplotlib renderer 转换为 QImage
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        qimage = QImage(buf, *canvas.get_width_height(), QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        plt.close(fig)  # 关闭图形，释放资源
        return pixmap
        # self.features_signal.emit(pixmap)

# if __name__ == '__main__':
#     data = {
#         'Feature_t': np.random.randn(2, 128),
#         'Feature_a': np.random.randn(2, 32),
#         'Feature_v': np.random.randn(2, 128),
#         'Feature_f': np.random.randn(2, 512),
#     }
#     th = Thread_Feature_Visualization(data)
#     th.run()
