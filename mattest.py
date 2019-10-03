import numpy as np
import cv2

one = np.ones((720,1280,3))
# print(one)
zero = np.zeros((720,1280,3))
# print(zero)
# result = np.insert(one,3, values = zero)
result = np.hstack((one, zero))
# cv2.imshow("result", result)

print(result)
print(result.shape)

key = ''
while key != 113:# for 'q' key
    cv2.imshow("zeros", zero)
    cv2.imshow("result", result)
    key = cv2.waitKey(5)
