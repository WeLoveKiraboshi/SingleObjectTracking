import cv2
import numpy as np
import time



def main(tracker = 'orb', save=False):
	max_inliers = 0
	min_inliers = 300
	sum_inliers = 0
	frame_count = 0
	elapsed_time = 0
	if tracker == 'orb':
		orb = cv2.ORB_create()  # ORB keypoints detector
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # key pts matcher
	elif tracker == 'akaze':
		orb = cv2.AKAZE_create() # AKAZE keypoint detector
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # key pts matcher
	elif tracker == 'brisk':
		orb = cv2.BRISK_create()
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # key pts matcher
	elif tracker == 'sift':
		orb = cv2.xfeatures2d.SIFT_create()
		matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # key pts matcher
	else:
		print('There is no option for keypoint detection : {}'.format(tracker))
		exit(0)

	if save == True:
		frame_rate = 45.0
		size = (640, 240) #	size = (640, 480)
		fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		writer = cv2.VideoWriter('./result/'+tracker+'_algorithm_b.mp4', fmt, frame_rate, size)


	src_keypts, src_desc, src_frame = list(), list(), np.array([])
	# hard-code bounding box in the 1st frame
	row, col, h, w = 24, 46, 170, 160  # (row, col) is the top-left corner of the bbox, h is bbox's height, w is bbox's width
	# NOTE: in image coordinate , the first coordinate is the row index, the 2nd coordinate is the column index
	# Create 4x2 matrix represents the pixel coordinate of 4 corners of the bbox defined by row, col, h, w. Each corners occupy a row.
	bbox = np.float32([[row, col], [row+h, col], [row+h, col+w], [row, col+w]]).reshape(-1, 1, 2)  # this reshape is make bbox compatible with function cv2.perspectiveTransform


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
			# first frame is being read, compute src keypoints and src descriptors
			# NOTE: these key points are computed inside the object of interest, not the entire image
			# define the binary image that is zero every where except for the points inside the bounding box of the
			# object of interest
			mask = np.zeros(gray_frame.shape)
			mask[row: row + h, col: col + w] = 1
			src_keypts, src_desc = orb.detectAndCompute(gray_frame, mask.astype(np.uint8))
			# draw the bbox on src_frame for displaying
			src_frame = cv2.polylines(frame, [np.int32(bbox)], True, 255, 3, cv2.LINE_AA)
			continue  # skip the code below

		# first frame was already read
		# from now on, key pts are matched with src_keypts to find the Homography that maps points in src_frame to
		# the current frame. By doing this, we can retrieve the bounding box of the object in the current frame.

		# find key pts and descriptor in the current frame. This time key points are detected on the entire image
		# because now we don't know where is object
		start = time.time()
		keypts, desc = orb.detectAndCompute(gray_frame, None)
		elapsed_time += time.time() - start

		# match key pts with src_keypts using there descriptors
		matches = matcher.match(src_desc, desc)

		# organize matched key pts into matrix
		matched_src_pts = np.float32([src_keypts[m.queryIdx].pt for m in matches]).reshape(-1, 2)  # shape (n_matches, 2)
		matched_pts = np.float32([keypts[m.trainIdx].pt for m in matches]).reshape(-1, 2)  # shape (n_matches, 2)

		M, inliers = cv2.findHomography(matched_src_pts, matched_pts, cv2.RANSAC,5.0)
		inliers_num = np.count_nonzero(inliers)
		if inliers_num < min_inliers:
			min_inliers = inliers_num
		if inliers_num > max_inliers:
			max_inliers = inliers_num
		sum_inliers += inliers_num
		frame_count += 1

		# map bbox from the 1st frame to the current frame
		current_bbox = cv2.perspectiveTransform(bbox, M)

		# display
		draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
		                   singlePointColor=None,
		                   matchesMask=inliers.ravel().tolist(),  # draw only inliers
		                   flags=2)
		if inliers_num >= 10:
			frame = cv2.polylines(frame, [np.int32(current_bbox)], True, 255, 3, cv2.LINE_AA)
			inliers_txt = str(inliers_num)
		else:
			inliers_txt = "lost"
		im_match = cv2.drawMatches(src_frame, src_keypts, frame, keypts, matches, None, **draw_params)
		cv2.putText(im_match, 'inliers:'+inliers_txt, (540, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.imshow('matches', im_match)
		if cv2.waitKey(50) & 0xFF == ord('q'):  # press "q" to end the program
			break
		#cv2.imshow('src_frame', src_frame)
		src_frame = frame
		src_keypts = keypts
		src_desc = desc
		bbox = current_bbox
		if save == True:
			writer.write(im_match)
		if frame_count == 250:
			cv2.imwrite('frame_algorithm(b)_'+str(frame_count)+'_'+str(tracker)+'.png', im_match)
			exit(0)


	cv2.destroyAllWindows()
	cap.release()
	if save == True:
		writer.release()
	print("================================================")
	print('Algorithm (b) : {}'.format(tracker))
	print('min inliers : {}'.format(min_inliers))
	print('max inliers : {}'.format(max_inliers))
	print('avg inliers : {}'.format(sum_inliers/frame_count))
	print('avg processing time : {}'.format(elapsed_time / frame_count))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='SOT')
	parser.add_argument('--s', default=False, type=bool, help='check when you wanna save output')
	parser.add_argument('--t', default="orb", type=str, help='algorithm of keypoint detection and description')
	args = parser.parse_args()
	main(args.t, args.s)
