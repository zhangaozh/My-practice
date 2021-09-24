from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import RankWarning
import xlrd

color = ['yellow', 'red', 'blue']

class Kmeans():

    def __init__(self, data, center_num=1):

        self.data = data

        self.center_num = center_num

    def classify(self, data, centers):

        classes = [[] for i in range(self.center_num)]
        
        sum_distances = 0

        for i in range(data.shape[0]):
            
            per_data_num = np.tile(data[i],(self.center_num,1))

            dis_mat = (per_data_num - centers)**2

            sum_distance = dis_mat.sum(axis=1)

            Index_sort = sum_distance.argsort()

            classes[Index_sort[0]].append(list(data[i]))

            sum_distances += sum_distance[Index_sort[0]]

        return classes, sum_distances
    
    def Updata_center(self, classes):

        centers = []

        for i in range(len(classes)):

            per_class = np.array(classes[i]) #classes is list

            center = per_class.sum(axis=0) / len(per_class)

            centers.append(center)

        return np.array(centers)

    def kmeans(self, centers, sum_distacne):
        #聚类
        classes, new_sumdistance = self.classify(self.data, centers)

        if sum_distacne == new_sumdistance:
            return
        #修改中心点
        New_center = self.Updata_center(classes)

        for i in range(len(New_center)):
            center = centers[i]
            plt.scatter(center[0], center[1], s=16*np.pi**2, marker='x', c=color[i])

        for i in range(len(classes)):
            per_class = classes[i]
            for c in per_class:
                plt.scatter(c[0], c[1], c=color[i])

        plt.show()

        self.kmeans(New_center, new_sumdistance)


def getData(xlsx):
    workbook = xlrd.open_workbook(xlsx)
    worksheet = workbook.sheet_by_index(0)
    nrows, ncols = worksheet.nrows, worksheet.ncols

    data = []

    for i in range(nrows):
        temp = [] #[12,13]
        for j in range(ncols):
            temp.append(worksheet.cell_value(i,j))
        data.append(temp)
    
    return np.array(data)

if __name__ == '__main__':
    data = getData('data.xlsx')
    centers = data[:3]
    K = Kmeans(data,len(centers))
    K.kmeans(centers, 0)