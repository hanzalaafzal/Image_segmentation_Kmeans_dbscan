import numpy as np
from tqdm import tqdm

class KMeansClustering:
    def __init__(self,K_clusters,max_iters,pixel_dict):
        self.K_clusters=K_clusters
        self.max_its = max_iters
        self.pixel_dict=pixel_dict
        self.pixel_vector_arr=pixel_dict['pixel_vector']

    '''
        function to initialize centroids
    '''

    def init_centroids(self,image_data):
        ctds = np.ndarray(shape=(self.K_clusters,5))
        for indx, ctd in enumerate(tqdm(ctds,desc='Generating initial Centroids')):
            ctds[indx] = np.random.uniform(self.pixel_dict['min'], self.pixel_dict['max'], 5)
        return ctds

    '''
        function to initialize centroids
    '''
    def euc_distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))


    def run_algo(self,ctds):

        its=0

        while(True):
            #copyinh previous generated centroids
            old_ctds=ctds.copy()

            _vector_scaled=self.pixel_dict['vector_scaled']

            #creating an array that contains the distances between each pixel and every centriod.

            for indx,pixel in enumerate(tqdm(_vector_scaled,desc='Calculating distances b/w ctd and pixel position')):
                ctd_distance=np.ndarray(shape=(self.K_clusters))
                for ctd_indx,ctd in enumerate(ctds):
                    ctd_distance[ctd_indx]=self.euc_distance(pixel.reshape(1,-1),ctd.reshape(1,-1))
                #storing minimum distance in pixel vector array
                self.pixel_vector_arr[indx]=np.argmin(ctd_distance)

            cluster_check=np.arange(self.K_clusters)
            cluster_empty=np.in1d(cluster_check,self.pixel_vector_arr)

            for indx,is_cluster in enumerate(tqdm(cluster_empty,desc='Checking if any cluster is empty')):
                if not is_cluster:
                    self.pixel_vector_arr[np.random.randint(len(self.pixel_vector_arr))] = indx


            for i in range(self.K_clusters):
                ctds[i]=np.mean(_vector_scaled[np.where(self.pixel_vector_arr==i)],axis=0)

            if(its == self.max_its):
                break

            print('\nIteration ',its+1,' completed')
            its += 1

        return self.pixel_vector_arr,ctds
