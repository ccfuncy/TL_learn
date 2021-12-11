from scipy.io.matlab.mio import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.metrics import accuracy_score
from scipy import io
import os


def load_data(folder,domain):
    '''数据集加载工作

    :param folder: 文件夹
    :type folder: string
    :param domain: 域
    :type domain: string
    '''
    data = io.loadmat(os.path.join(folder,domain+'_fc7.mat'))
    return data['fts'],data['labels']

def knn_classify(Xs,Ys,Xt,Yt,k=1):
    '''knn分类器

    :param Xs: 源域特征
    :type Xs: array
    :param Ys: 源域标签
    :type Ys: array
    :param Xt: 目标域特征
    :type Xt: array
    :param Yt: 目标域标签
    :type Yt: array
    :param k: k近邻的超参数 defaults to 1
    :type k: int, optional
    '''
    Ys = Ys.ravel()
    Yt = Yt.ravel()
    # model = KNeighborsClassifier(n_neighbors=k).fit(Xs,Ys)
    model = make_pipeline(Normalizer(),KNeighborsClassifier(n_neighbors=k)).fit(Xs,Ys)

    acc = accuracy_score(model.predict(Xt),Yt)
    print("Accurary using KNN: {:.2f}%".format(acc*100))

folder = '../../data/office31-decaf'
src_domain = 'amazon'
tar_domain = 'webcam'

Xs,Ys=load_data(folder,src_domain)
Xt,Yt=load_data(folder,tar_domain)

print('Source:',src_domain,Xs.shape,Ys.shape)
print('Target:',tar_domain,Xt.shape,Yt.shape)


print([knn_classify(Xs,Ys,Xt,Yt,i) for i in range(1,100)])
