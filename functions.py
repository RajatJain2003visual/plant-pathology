import cv2
import numpy as np
import skimage

def mean_brightness(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  return np.mean(v)

def std_brightness(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  return np.std(v)

def mean_saturation(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  return np.mean(s)

def std_saturation(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  return np.std(s)

def yellow_area_fraction(img):
  lower_yellow = np.array([20, 100, 100])
  upper_yellow = np.array([40, 255, 255])

  yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
  yellow_pixels = np.sum(yellow_mask > 0)
  total_pixels = yellow_mask.size
  return yellow_pixels / total_pixels

def lbp_hist(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  lbp = skimage.feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
  lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
  return lbp_hist

def number_of_blobs(img):
  lower_yellow = np.array([20, 100, 100])
  upper_yellow = np.array([40, 255, 255])

  yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(yellow_mask, connectivity=8)
  blob_areas = stats[1:, cv2.CC_STAT_AREA]
  num_blobs = len(blob_areas)
  avg_blob_size = np.mean(blob_areas) if num_blobs > 0 else 0
  feature = []
  feature.append(num_blobs)
  feature.append(avg_blob_size)
  return feature

def extract_features(img_path):
#   img_path = os.path.join('images', img_path) + ".jpg"
  img = cv2.imread(img_path)
  img = cv2.resize(img, (500, 500))
  features = []
  features.append(mean_brightness(img))
  features.append(std_brightness(img))
  features.append(mean_saturation(img))
  features.append(std_saturation(img))
  features.append(yellow_area_fraction(img))
  features.extend(lbp_hist(img))
  features.extend(number_of_blobs(img))
  features = np.array(features)
  return features