from PIL import Image
import numpy as np
import math
from tqdm import tqdm

class DBSCAN:

    def __init__(self,eps,min_pts,pixel_data):
        self.eps=eps
        self.min_pts=min_pts
        self.clusters=0
        self.pixel_data=pixel_data
        self.points_arr = []

        for y in tqdm(range(len(self.pixel_data['image_array'])),desc='Generating points array'):
            self.points_arr.append([])
            for x in range(len(self.pixel_data['image_array'][0])):
                self.points_arr[y].append([y,x,self.pixel_data['image_array'][y][x],"unvisited"])

    '''
        function to calculate euclidean distance (L2)
    '''

    def euc_distance(self,x,y):
        _distances = []
        for i in range(len(x[2])):
            _distances.append(math.pow(y[2][i] - x[2][i],2))
        return math.sqrt(sum(_distances))


    '''
        function to find neighbors of current point
    '''
    def find_neighbors(self,crr_point):
        _neighbors=[]
        for y in range(len(self.points_arr)):
            for x in range(len(self.points_arr[0])):
                if self.euc_distance(crr_point,self.points_arr[y][x]) <= self.eps:
                    _neighbors.append(self.points_arr[y][x]) #since the neighboring point is in epsilon distance, it is considered as neighbor
        return _neighbors

    def generate_clusters(self):
        for y in range(len(self.pixel_data['image_array'])):
            for x in range(len(self.pixel_data['image_array'][0])):

                #checking if point is already visited
                if(self.points_arr[y][x][-1] != 'unvisited'):
                    continue

                #finding neighbors
                neighbors=self.find_neighbors(self.points_arr[y][x])

                #checking the noise points i.e. outside the circle distance
                if len(neighbors) < self.min_pts:
                    self.points_arr[y][x][-1] = 'noise'
                    continue


                #once the noise is checked, we have the clusters formed
                self.clusters = self.clusters + 1
                self.points_arr[y][x][-1]= str(self.clusters) #putting the number of cluster

                print('\nCluster '+str(self.clusters)+' formed')
                print('\nNeighbors of cluster: ')
                print(neighbors)


                if self.points_arr[y][x] in neighbors:
                    neighbors.remove(self.points_arr[y][x])

                for innerpoint in neighbors:
                    if innerpoint[-1] == 'noise':
                        #if the neighbor was previously marked as noise by other cluster, it becomes part of this cluster
                        self.points_arr[innerpoint[0]][innerpoint[1]][-1] = str(self.clusters)

                    if innerpoint[-1] != 'unvisited':
                        #if the it was already marked as neighbor of other core point or a part of cluster, it stays same
                        continue
                    #marking the neighboring point as cluster member
                    self.points_arr[innerpoint[0]][innerpoint[1]][-1] = str(self.clusters)

                    neighbors_inner=self.find_neighbors(innerpoint)
                    #checking neighbors of inner points of the neighbors of core point
                    if len(neighbors_inner) >= self.min_pts:
                        neighbors.append(neighbors_inner)
    '''
        function to generate array with cluster numbers
    '''
    def gen_clusters_numbers(self):
        clusters_numbers = []
        for y in range(len(self.pixel_data['image_array'])):
            for x in range(len(self.pixel_data['image_array'][0])):
                if self.points_arr[y][x][-1] not in clusters_numbers:
                    clusters_numbers.append(self.points_arr[y][x][-1])

        return clusters_numbers

    def calc_vectors(self,clusters_numbers):
        #calculating averages of clusters
        clusters_averages=[]
        for i in clusters_numbers:
            n=0
            _temp_vector=[0]*len(self.points_arr[0][0][2])
            for y in range(len(self.pixel_data['image_array'])):
                for x in range(len(self.pixel_data['image_array'][0])):
                    if self.points_arr[y][x][-1] == i:
                        for j in range(len(self.points_arr[y][x][2])):
                            _temp_vector[j]=_temp_vector[j]+self.points_arr[y][x][2][j]
                        n=n+1
            #Checking zero division error
            for d in range(len(_temp_vector)):
                if _temp_vector[d] != 0:
                    _temp_vector[d] = _temp_vector[d]/n
            clusters_averages.append(_temp_vector)

        #Building Array with Clusters and chaning the averages of clusters with initial values
        clusters_vectors=[]
        for y in range(len(self.points_arr)):
            clusters_vectors.append([])
            for x in range(len(self.points_arr[0])):
                clusters_vectors[y].append(clusters_averages[clusters_numbers.index(self.points_arr[y][x][-1])])
        return clusters_vectors
