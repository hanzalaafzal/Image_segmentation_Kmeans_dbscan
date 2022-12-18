from PIL import Image
import numpy as np
import os
from pathlib import Path
from sklearn import preprocessing
from tqdm import tqdm


class ImageToPixel:

    def __init__(self,image_path):
        self.file=image_path
        _file=os.path.basename(image_path)

        for i in range(1,1000):
            _output_folder='output_'+os.path.splitext(_file)[0]+'_'+str(i)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
                self.output_folder=_output_folder
                break

    def get_base_path():
        return self.output_folder

    def resize_image(self,image_obj,height,width):

        size=(width,height)
        image_obj=image_obj.resize(size)
        return image_obj

    def image_vector_kmeans(self,resize=False,height=0,width=0):
        image=Image.open(self.file)

        if(resize==True):
            image=self.resize_image(image,height,width)

        _x=image.size[0]
        _y=image.size[1]

        #converting pixels into np array with 5 features
        vector_arr=np.ndarray(shape=(_x * _y,5), dtype=float)

        pixelvector_arr=np.ndarray(shape=(_x * _y), dtype=int)


        #Gathering vectors of all attributes with data from input image
        #creating a list of vectors, each one representing the RGB values for a single pixel.
        for y in tqdm(range(0, _y),desc='Generating vectors of all attibutes with pixels and coordinates'):
            for x in range(0, _x):
                xy = (x, y)
                rgb = image.getpixel(xy)
                vector_arr[x + y * _x, 0] = rgb[0]
                vector_arr[x + y * _x, 1] = rgb[1]
                vector_arr[x + y * _x, 2] = rgb[2]
                vector_arr[x + y * _x, 3] = x
                vector_arr[x + y * _x, 4] = y

        #Standardize the vectors of our features
        vector_scaled = preprocessing.normalize(vector_arr)

        #Extracting minimum and maximum values from scaled vector data
        self._min_scaled = np.amin(vector_scaled)
        self._max_scaled = np.amax(vector_scaled)

        self.pixel_dict={'min':self._min_scaled,'max':self._max_scaled,'vector_scaled':vector_scaled,'pixel_vector':pixelvector_arr,'image_obj':image}
        return self.pixel_dict



    def convert_vector_image_kmeans(self,processed_array,ctds,image_obj,k,its):
        image_array=np.asarray(image_obj)
        image_array_cp=image_array.copy()
        # image_array.setflags(write=1)
        for indx,pixel_vector in enumerate(processed_array):
            x, y= indx // image_obj.size[0], indx % image_obj.size[0]
            image_array_cp[x, y] = np.round(ctds[pixel_vector][:3] * 255)

        new_image=Image.fromarray(image_array_cp)

        _image_name='Kmeans_'+k+its
        new_image.save(self.output_folder+'/'+_image_name+'_K.jpg')


    def image_vector_dbscan(self,resize=False,height=0,width=0):
        image=Image.open(self.file)

        if(resize==True):
            image=self.resize_image(image,int(height),int(width))

        image_arr=np.asarray(image)
        self.pixel_dict={'image_obj':image,'image_array':image_arr}
        return self.pixel_dict

    def convert_vector_image_db(self,vectors,eps,min_pts):
        image_arr=np.asarray(vectors)
        image_arr_cp=np.uint8(image_arr)
        image_obj=Image.fromarray(image_arr_cp)
        image_obj.save(self.output_folder+'/'+eps+'_'+min_pts+'_db.jpg')

    def __del__(self):
        print('-----------------------------Image Saved----------------------------------')
