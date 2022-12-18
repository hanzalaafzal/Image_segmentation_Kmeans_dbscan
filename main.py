
from image2px import ImageToPixel
from kmeans import KMeansClustering
from dbscan import DBSCAN
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Image segmentation script')
parser.add_argument('-i','--image',metavar='',required=True,help='Input image')
parser.add_argument('-a','--algo',metavar='',required=True,help='Algorithm i.e. either Kmeans or dbscan')
parser.add_argument('-r','--resize',action='store_true',help='Image resize flag. Recommended with dbscan')


args=parser.parse_args()

i2p=ImageToPixel(args.image)


if(args.resize):
    RESIZE_FLAG=True
    height=input('Height > ')
    while not height:
        height=input('Height > ')
    width=input('Width > ')
    while not width:
        width=input('Width > ')
else:
    RESIZE_FLAG=False
    height=0
    width=0


if args.algo == 'kmeans':
    k_clusters=input('No of Clusters > ')
    while not k_clusters:
        k_clusters=input('No of Clusters > ')

    max_its=input('Max iterations > ')
    while not max_its:
        max_its=input('Max iterations > ')

    pixel_data=i2p.image_vector_kmeans(RESIZE_FLAG,int(height),int(width))

    kmeans_obj=KMeansClustering(int(k_clusters),int(max_its),pixel_data)
    initial_centroid=kmeans_obj.init_centroids(pixel_data)
    p_image_arr,p_ctds=kmeans_obj.run_algo(initial_centroid)
    i2p.convert_vector_image_kmeans(p_image_arr,p_ctds,pixel_data['image_obj'],k_clusters,max_its)

elif args.algo == 'dbscan':
    epsilon=input('Epsilon > ')
    while not epsilon:
        epsilon=input('Epsilon > ')

    minimum_points=input('Minimum Points > ')
    while not minimum_points:
        minimum_points=input('Minimum Points > ')

    pixel_data=i2p.image_vector_dbscan(RESIZE_FLAG,height,width)
    dbscan_obj=DBSCAN(int(epsilon),int(minimum_points),pixel_data)

    dbscan_obj.generate_clusters()

    cluster_numbers=dbscan_obj.gen_clusters_numbers()
    cluster_vectors=dbscan_obj.calc_vectors(cluster_numbers)
    i2p.convert_vector_image_db(cluster_vectors,epsilon,minimum_points)
