import numpy as np
import matplotlib.pyplot as plt

from kalmanfilter import KalmanFilter

plt.ion() # interactive mode will be on <-> plot.ioff()
plt.figure()

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS =20

kf = KalmanFilter(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

means = []
covs = []

for step in range(NUM_STEPS):
    means.append(kf.mean)
    covs.append(kf.cov)
    
    kf.predict(dt=DT)

plt.subplot(211) #or plt.subplot(2,1,1)
plt.title('Position')
plt.plot([mean[0] for mean in means], 'r')
#Bounds(2-sigma, i.e., 95% confidence interval)
plt.plot([mean[0] - 2*np.sqrt(cov[0,0]) for mean, cov in zip(means, covs)],'r--')
plt.plot([mean[0] + 2*np.sqrt(cov[0,0]) for mean, cov in zip(means, covs)],'r--')

plt.subplot(212)
plt.title('velocity')
plt.plot([mean[1] for mean in means], 'r')
#Bounds(2-sigma, i.e., 95% confidence interval)
plt.plot([mean[1] - 2*np.sqrt(cov[1,1]) for mean, cov in zip(means, covs)],'r--')
plt.plot([mean[1] + 2*np.sqrt(cov[1,1]) for mean, cov in zip(means, covs)],'r--')


plt.show()
x = plt.ginput(1)
print(x)