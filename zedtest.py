import numpy as np
import pyzed.sl as sl
import cv2

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1


def main():
    print("Running...")
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    depth_map = sl.Mat()
    depth_fordisp = sl.Mat()
    disparity_map = sl.Mat()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
#             cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            cam.retrieve_image(depth_fordisp, sl.VIEW.VIEW_DEPTH)
#             cam.retrieve_measure(depth_map, sl.MEASURE.MEASURE_DEPTH)
#             cam.retrieve_measure(disparity_map, sl.MEASURE.MEASURE_DISPARITY)
#             depth_value = depth_map.get_value(100,100)
#             cv2.imshow("ZED", mat.get_data())
            cvdata = depth_fordisp.get_data()
            noalpha = cv2.cvtColor(cvdata, cv2.COLOR_RGBA2RGB)
#             for row in range(len(noalpha)):
#                 for col in range(len(noalpha[row])):
#                     for val in range(len(noalpha[row, col])):
#                         if noalpha[row,col,val] == 0:
#                             noalpha[row,col] = [0,0,255]
#                             cv2.imshow("RED", noalpha)
            noalpha[noalpha[100,100] == [0,0,0]] = np.arange(720)
#             print(noalpha[:,:,2])
#             noalpha[noalpha < 144] = 255

            cv2.imshow("RED", noalpha)
#             print(noalpha.shape)
#             print("cvdata(4):", cvdata, "noalpha(3):", noalpha)
#             print(cvdata)
#             rbg = cv2.cvtColor(cvdata, cv2.COLOR_GRAY2BGR)
#             cv2.imshow("Depth", depth_fordisp.get_data())
#             cv2.imshow("RGB", rbg)
#             cv2.imshow("Measured_Depth", depth_map.get_data())
#             cv2.imshow("Disparity", disparity_map.get_data())
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
