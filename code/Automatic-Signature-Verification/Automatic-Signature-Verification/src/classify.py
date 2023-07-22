import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class Classifier:
    
    def __init__(self, threshold=0.11):
        self.threshold = threshold
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, extended=1)

    def load_image(self, path):
        with Image.open(path).convert('L') as img:
            return np.array(img)

    def extract_keypoints(self, img):
        kp, des = self.surf.detectAndCompute(img, None)
        pts = np.array([[int(p.pt[0]), int(p.pt[1])] for p in kp])
        return des, pts
    
    def classify_sig(self, img_path, stable_descs, plot=False):
        img = self.load_image(img_path)
        des, pts = self.extract_keypoints(img)
        tot_des = des.shape[0]
        matches = 0
        stable_kp = []
        unstable_kp = []
        for i, desc in enumerate(des):
            min_dist = np.min(np.linalg.norm(stable_descs - desc, axis=1))
            if min_dist <= self.threshold:
                matches += 1
                stable_kp.append(pts[i])
            else:
                unstable_kp.append(pts[i])
        stable_kp = np.array(stable_kp)
        unstable_kp = np.array(unstable_kp)
        if plot:
            self.plot(img, stable_kp, unstable_kp)
        return matches / tot_des
    
    def classify(self, query_path, stable_descs, plot=False):
        queries = os.listdir(query_path)
        percentage_match = []
        for i, query_name in enumerate(queries):
            img_path = os.path.join(query_path, query_name)
            match_percentage = self.classify_sig(img_path, stable_descs, plot=plot)
            percentage_match.append(match_percentage)
        return np.array(percentage_match)

    def plot(self, img, stable_kp, unstable_kp):
        fig = plt.figure(figsize=(14, 14), dpi=60)
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax.imshow(img, cmap='gray')
        ax1.imshow(img, cmap='gray')
        ax.scatter(stable_kp[:,0], stable_kp[:,1], color='green', label='stable')
        ax1.scatter(unstable_kp[:,0], unstable_kp[:,1], color='red', label='unstable')
        ax.axis('off')
        ax1.axis('off')
        ax.set_title('Stable')
        ax1.set_title('Unstable')
        plt.legend()
        plt.show()