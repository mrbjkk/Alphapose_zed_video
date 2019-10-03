import sys
import pyzed.sl as sl
import cv2


def main():

#     if len(sys.argv) != 2:
#         print("Please specify path to .svo file.")
#         exit()

#     filepath = sys.argv[1]
    filepath = '/home/yurik/Documents/Program/Alphapose_zed_video/testdata/high_accuracy/HD720_30.svo'
    print("Reading SVO file: {0}".format(filepath))

    init = sl.InitParameters(svo_input_filename=filepath,svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    lena = cv2.imread('/home/yurik/Pictures/Lena.png')

    key = ''
    print("  Save the current image:     s")
    print("  Quit the video reading:     q\n")
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_DEPTH)
            display = mat.get_data()
            print(display.shape[2])
            display = cv2.cvtColor(display, cv2.COLOR_RGBA2RGB)
#             print(display.shape)
#             lena = cv2.resize(lena, (display.shape[1], display.shape[0]))
#             lena = display.copy()
#             cv2.line(lena,(10,10),(200,200),(0,255,0),3)
#             cv2.addWeighted(display, 0.5, lena, 1.5, 3, display)
            cv2.imshow("ZED", display)
#             cv2.imshow("Lena", lena)
            key = cv2.waitKey(1)
#             saving_image(key, mat)
        else:
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()

#     print_camera_information(cam)
#     saving_depth(cam)
#     saving_point_cloud(cam)

    cam.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
