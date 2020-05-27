import cv2
import glob

#code snipped from https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

def makevideo(layers_list):
    img_array = []
    for filename in glob.glob('observe/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('observe/grid_search_vids/plasticity_'+str(layers_list)+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

