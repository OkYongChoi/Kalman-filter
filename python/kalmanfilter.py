import numpy as np
class KalmanFilter:
    """Class for Kalman Filter"""
    def __init__(self, initial_x: float,
                       initial_v: float,
                       accel_variance: float) -> None:
        #mean of state Gaussian RV
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance
        
        #covariance of state Gaussian RV
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        """
        x = F*x
        P = F P Ft + G sigma^2 Gt     
        """
        F = np.array([[1,dt], [0,1]])
        new_x = F @ self._x
        
        G = np.array([[0.5 * dt**2], [dt]])
        new_P = F @ self._P @ F.T + G * self._accel_variance @ G.T

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        """
        y = z - H x
        S = H P Ht + R
        K = P Ht S^-1
        x = x + K y
        P = (I -K H) * P

        It's not a very stable formula but for now we leave it like this.
        """

        # For the sake of consistency, convert the values and variance into np.arrays
        z = np.array([meas_value])
        R = np.array([meas_variance])
        
        H = np.array([1, 0]).reshape((1,2))

        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        
        K = self._P @ H.T @ np.linalg.inv(S)

        new_x = self._x + K @ y
        new_P = (np.eye(2) - K @ H) @ self._P


        self._x = new_x
        self._P = new_P
        








    @property
    def pos(self) -> float:
        return self._x[0]
    
    @property
    def vel(self) -> float:
        return self._x[1]

    @property
    def mean(self) -> np.array:
        return self._x
    
    @property
    def cov(self) -> np.array:
        return self._P

    @pos.setter
    def pos(self, x) -> float:
        self._x[0]=x

    @vel.setter
    def vel(self,v) -> float:
        self._x[1]=v