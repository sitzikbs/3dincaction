import cv2
import os

image_folder = '/home/sitzikbs/PycharmProjects/dfaust/figures/log/gradcam/hips/000006/front'
video_name = os.path.join(image_folder, 'dfaust_gradcam_hips_female.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
images.sort()
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
images.reverse()
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

os.system('ffmpeg -an -i {} -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 {}'.format(video_name,
                                                                                                     video_name[:-4] + '_html5' + video_name[-4:]))