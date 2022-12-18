## Image Segmentation
Image segmentation is the process of taking a digital image and dividing it into subgroups called segments, thereby reducing the overall complexity of the image and enabling the analysis and processing of each segment. If we delve into image segmentation further, we see that a segmentation image is all about assigning particular labels to pixels to identify objects, people, and other important elements. More details [Here](https://mindy-support.com/news-post/what-is-image-segmentation-the-basics-and-key-techniques).

#### The following program performs image segmentation using [Kmeans](https://databasecamp.de/en/ml/k-means-clustering) clustering and [DBSCAN](https://www.mygreatlearning.com/blog/dbscan-algorithm/).
--------------------------
Steps to install and run

Download the above scripts (main.py,kmeans.py, dbscan.py, image2px.py, requirements.txt)

1. Install the required libraries using pip

```
> pip install -r requirements.txt
```
2. Run help command

```
> python main.py -h

Output:
  -h, --help     show this help message and exit
  -i , --image   Input image
  -a , --algo    Algorithm i.e. either Kmeans or dbscan
  -r, --resize   Image resize flag. Recommended with dbscan
```
3. Run kmeans clustering

```
> python main.py -i image_path/image.jpg -a kmeans -r

Height > 400

Width > 400

No of Clusters > 10

Max iterations > 100

```

4. Run DBSCAN

```
> python main.py -i image_path/image.jpg -a dbscan -r

Height > 100

Width > 100

Epsilon > 10

Minimum Points > 100

```
5. Output

Image will be stored in output_{image_filename}/10_100_K.jpg or output_{image_filename}/10_100_db.jpg

Use -r flag to resize image. Recommended for DBSCAN algorithm for fast processing.

----------------------
## Authors
1. Madiha Shaikh (Matriculation: 5001778)

2. Muhammad Hanzala Afzal (Matriculation: 5002240)
