import os
import cv2
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

class ReferenceDatabase:
    
    def __init__(self, path):
        self.path = path
        self.img = None
        self.surf_db = None
        self.stableDB = None
        self.unstableDB = None

    def load_images(self):
        # Load images from a given path using PIL
        img = []
        file_names = os.listdir(self.path)
        for img_name in file_names:
            l = len(img_name)
            if img_name[l - 4: l] == ".png":
                im = Image.open(os.path.join(self.path, img_name)).convert('L')
                img.append(np.array(im))
        self.img = np.array(img)

    def generate_surf_db(self):
        # Perform SURF for each Reference Image
        surf_db = []
        surf_kps = []
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 1000, extended = 1)
        for i in range(self.img.shape[0]):
            kp, des = surf.detectAndCompute(self.img[i], None)
            surf_db.append(des)
            surf_kps.append(kp)
        self.surf_db = np.array(surf_db)
        self.surf_kps = np.array(surf_kps)

    def generate_kp_db(self):
        # Create stable and unstable keypoints databases
        stableDB = []
        unstableDB = []
        stableKp = []
        unstableKp = []
        
        for i in range(self.img.shape[0]):
            # Generate LOO database
            indexes = list(range(self.img.shape[0]))
            indexes.remove(i)
            tmp_db = np.concatenate(self.surf_db[indexes])

            # Find minimum distance for each keypoint
            sum_dist = 0
            distances = []
            for keypoint in self.surf_db[i]:
                min_dist = np.min(np.linalg.norm(tmp_db - keypoint, axis=1))
                sum_dist += min_dist
                distances.append(min_dist)
            avg_dist = sum_dist / self.surf_db[i].shape[0]

            # Append keypoints less than average distance
            stable_indexes = np.where(distances <= avg_dist)
            stableDB.extend(list(self.surf_db[i][stable_indexes]))
            stableKp.append([self.surf_kps[i][j] for j in stable_indexes[0]])
            
            # Append keypoints greater than average distance
            unstable_indexes = np.where(distances > avg_dist)
            unstableDB.extend(list(self.surf_db[i][unstable_indexes]))
            unstableKp.append([self.surf_kps[i][j] for j in unstable_indexes[0]])
            
        self.stableDB = np.array(stableDB)
        self.unstableDB = np.array(unstableDB)
        
        self.stableKp = stableKp
        self.unstableKp = unstableKp
    
    def kp_DB(self):
        # Create the Reference Database for both stable and unstable keypoints
        self.load_images()
        self.generate_surf_db()
        self.generate_kp_db()
        return self.stableDB, self.unstableDB

    def save_DB(self, filename):
        # Save the Reference Database
        with open(filename, 'wb') as f:
            pickle.dump(self.stableDB, f)

    def load_DB(self, filename):
        # Load the Reference Database
        with open(filename, 'rb') as f:
            self.stableDB = pickle.load(f)

    def visualize_stable(self):
        # Visualize the stable keypoints
        for i in range(self.img.shape[0]):
            plt.figure(figsize = (8, 8), dpi = 60)
            tmp = cv2.drawKeypoints(self.img[i],self.stableKp[i],None,color = (0,255,0))
            plt.imshow(tmp)
            plt.axis('off')
            plt.title('Stable Keypoints')
            plt.show()
            
    def visualize_unstable(self):
        # Visualize the unstable keypoints
        for i in range(self.img.shape[0]):
            plt.figure(figsize = (8, 8), dpi = 60)
            tmp = cv2.drawKeypoints(self.img[i],self.unstableKp[i],None,color = (255,0,0))
            plt.imshow(tmp)
            plt.axis('off')
            plt.title('Unstable Keypoints')
            plt.show()
