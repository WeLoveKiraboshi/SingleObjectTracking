import cv2
import numpy as np
import time


def calc_center(point):
    #points : (4, 1, 2)
    center = np.mean(point, axis = 0).reshape(2)
    return center #(row, col)






def main(save=False):
    #max_inliers = 0
    #min_inliers = 300
    sum_inliers = 0
    frame_count = 0
    #elapsed_time = 0

    tracker = 'orb'
    orb = cv2.ORB_create()  # ORB keypoints detector
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # key pts matcher

    if save == True:
        frame_rate = 45.0
        size = (640, 240) #	size = (640, 480)
        # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('./result/'+tracker+'_algorithm_a.mp4', fmt, frame_rate, size)

    src_keypts, src_desc, src_frame = list(), list(), np.array([])
    # hard-code bounding box in the 1st frame
    row, col, h, w = 24, 46, 170, 160  # (row, col) is the top-left corner of the bbox, h is bbox's height, w is bbox's width
    # NOTE: in image coordinate , the first coordinate is the row index, the 2nd coordinate is the column index
    # Create 4x2 matrix represents the pixel coordinate of 4 corners of the bbox defined by row, col, h, w. Each corners occupy a row.
    bbox = np.float32([[row, col], [row+h, col], [row+h, col+w], [row, col+w]]).reshape(-1, 1, 2)  # this reshape is make bbox compatible with function cv2.perspectiveTransform

    # State vector [x, y, theta, x_dot, y_dot] #orientation of bounding box
    # Measurement vector [x, y] center of bounding box
    StateSize = 5
    MeasSize = 2
    dt = 1 # consider the Constant velocity linear motion
    kalman = cv2.KalmanFilter(StateSize, MeasSize)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 0, dt, 0],
                                        [0, 1, 0, 0, dt],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 1]], np.float32) * 0.03
    measurement = np.array([[(2*row)/h], [(2*col)/h]], np.float32) #(2, 1)
    prediction = np.array([[(2*row)/h], [(2*col)/h], [0], [0], [0]], np.float32) #(5, 1)
    print("\nObservation in image: BLUE")
    print("Prediction from Kalman: GREEN\n")
    cv2.namedWindow("Kalman")


    # open video
    cap = cv2.VideoCapture('video1.mp4')
    # main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Video has ended. Exitting')
            break

        # convert frame to gray scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not src_keypts:
            mask = np.zeros(gray_frame.shape)
            mask[row: row + h, col: col + w] = 1
            src_keypts, src_desc = orb.detectAndCompute(gray_frame, mask.astype(np.uint8))
            # draw the bbox on src_frame for displaying
            src_frame = cv2.polylines(frame, [np.int32(bbox)], True, 255, 3, cv2.LINE_AA)
            continue  # skip the code below

        #start = time.time()
        keypts, desc = orb.detectAndCompute(gray_frame, None)
        #elapsed_time += time.time() - start
        # match key pts with src_keypts using there descriptors
        matches = matcher.match(src_desc, desc)
        # organize matched key pts into matrix
        matched_src_pts = np.float32([src_keypts[m.queryIdx].pt for m in matches]).reshape(-1,2)  # shape (n_matches, 2)
        matched_pts = np.float32([keypts[m.trainIdx].pt for m in matches]).reshape(-1, 2)  # shape (n_matches, 2)

        M, inliers = cv2.findHomography(matched_src_pts, matched_pts, cv2.RANSAC, 3.0)
        inliers_num = np.count_nonzero(inliers)
        sum_inliers += inliers_num
        frame_count += 1

        # map bbox from the 1st frame to the current frame
        current_bbox = cv2.perspectiveTransform(bbox, M) #(4, 1, 2)
        current_center = calc_center(current_bbox) #observation
        kalman.correct(current_center)
        prediction = kalman.predict() #(StateSize, 1)
        pred_x = prediction[0]
        pred_y = prediction[1]
        pred_theta = prediction[2]
        frame = cv2.rectangle(frame,(int((pred_y-w/2)), int((pred_x-h/2))),
                              (int((pred_y+w/2)), int((pred_x+h/2))),(0,255,0),2)
        if cv2.waitKey(50) & 0xFF == ord('q'):  # press "q" to end the program
            break
        cv2.imshow('frame', frame)
        if save == True:
            writer.write(im_match)




    cv2.destroyAllWindows()
    cap.release()
    if save == True:
        writer.release()
    print("================================================")
    print('Algorithm (a) : {}'.format(tracker))
    print('min inliers : {}'.format(min_inliers))
    print('max inliers : {}'.format(max_inliers))
    print('avg inliers : {}'.format(sum_inliers / frame_count))
    print('avg processing time : {}'.format(elapsed_time / frame_count))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SOT')
    parser.add_argument('--s', default=False, type=bool,
						help='check when you wanna save output')
    args = parser.parse_args()
    main(args.s)
