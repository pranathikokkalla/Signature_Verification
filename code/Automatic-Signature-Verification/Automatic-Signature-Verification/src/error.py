import numpy as np
import matplotlib.pyplot as plt

class ErrorPlotter:
    def __init__(self, genuine_matches, disguise_matches, sim_matches, theta_range=np.arange(0.005, 0.05, 0.0005)):
        self.genuine_matches = genuine_matches
        self.disguise_matches = disguise_matches
        self.sim_matches = sim_matches
        self.far = []
        self.frr = []
        self.theta_range = theta_range
        self.eer = None
        
    def find_far_frr(self):
        for theta in self.theta_range:
            gen = np.where(self.genuine_matches <= theta)
            disg = np.where(self.disguise_matches <= theta)
            sim = np.where(self.sim_matches > theta)
            self.frr.append((gen[0].shape[0] + disg[0].shape[0]) / (self.genuine_matches.shape[0] + self.disguise_matches.shape[0]) * 100)
            self.far.append(sim[0].shape[0] / self.sim_matches.shape[0] * 100)
        
        self.diff = np.abs(np.array(self.frr) - np.array(self.far))
        self.idx = np.where(self.diff == min(self.diff))[0][0]
        self.eer = self.far[self.idx]
        return self.far, self.frr, self.eer
    
    def plot_error(self, threshold):
        if self.eer is None:
            self.find_far_frr()
            
        plt.figure(figsize = (8, 8))
        plt.plot(self.theta_range, self.far, color = 'red')
        plt.plot(self.theta_range, self.frr, color = 'blue')
        plt.scatter(self.theta_range[np.argmin(np.abs(np.array(self.frr) - np.array(self.far)))],
                    (self.far[self.idx]+self.far[self.idx-1])/2, color = 'black', marker = 'o', s=50)
        plt.title('error rate')
        plt.legend(['FAR', 'FRR', 'EER'])
        plt.xlabel('Theta')
        plt.ylabel('Error rate')
        plt.show()
