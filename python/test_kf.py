from kalmanfilter import KalmanFilter
import unittest
import numpy as np

class TestKF(unittest.TestCase):

    def test_can_construct_with_x_and_v(self):
        x = 0.2
        v = 2.0


        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)
        self.assertAlmostEqual(kf.pos,x)
        self.assertAlmostEqual(kf.vel,v)

    def test_can_predict(self):
        x = 0.2
        v = 2.0

        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)
        kf.predict(0.3)

    def test_after_calling_predict_x_and_P_are_of_right_shaoe(self):
        x = 0.2
        v = 2.0

        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)
        kf.predict(0.1)

        self.assertEqual(kf.cov.shape, (2, 2))
        self.assertEqual(kf.mean.shape, (2, ))

    def test_after_calling_predict_increases_state_uncertainty(self):
        """ 
        Everytime I predict my uncertainty estimate should increse.
        More accurately, differential entropy is the determinant of matrix P
        """
        x = 0.2
        v = 2.0

        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after= np.linalg.det(kf.cov)

            self.assertGreater(det_after, det_before)

            print(det_before, det_after) # If you would like to see the content of print() in the unittest, type -s after pytest ., i.e., pytest . -s

    def test_calling_update_method_does_not_crash(self):
        x = 0.2
        v = 2.0

        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)

        kf.update(meas_value=0.1, meas_variance=0.1)

    def test_calling_update_decreases_state_uncertainty(self):
        """
        After calling update, for the analogous reason, uncertainty should decreas the uncertainty.
        update allows us to bound the uncertainity of our state.
        """
        x = 0.2
        v = 2.0

        kf = KalmanFilter(initial_x=x, initial_v=v, accel_variance=1.5)

        det_before = np.linalg.det(kf.cov)
        kf.update(meas_value=0.1, meas_variance=0.01)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after, det_before)