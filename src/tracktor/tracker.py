import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.autograd import Variable

import cv2
import pdb
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from frcnn.model.nms_wrapper import nms
from frcnn.model import test

from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes

transforms = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


class Tracker():
	"""The main tracking file, here is where magic happens."""
	# only track pedestrian
	cl = 1

	def __init__(self, obj_detect, obj_detect_head ,reid_network, tracker_cfg):
		self.obj_detect = obj_detect
		self.obj_detect_head = obj_detect_head
		self.reid_network = reid_network
		self.detection_person_thresh = tracker_cfg['detection_person_thresh']
		self.regression_person_thresh = tracker_cfg['regression_person_thresh']
		self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
		self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
		self.public_detections = tracker_cfg['public_detections']
		self.inactive_patience = tracker_cfg['inactive_patience']
		self.do_reid = tracker_cfg['do_reid']
		self.max_features_num = tracker_cfg['max_features_num']
		self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
		self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
		self.do_align = tracker_cfg['do_align']
		self.motion_model = tracker_cfg['motion_model']

		self.warp_mode = eval(tracker_cfg['warp_mode'])
		self.number_of_iterations = tracker_cfg['number_of_iterations']
		self.termination_eps = tracker_cfg['termination_eps']

		self.reset()

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.pos = t.last_pos
		self.inactive_tracks += tracks

	def add(self, new_det_pos, new_det_scores, new_det_features):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(new_det_pos[i].view(1,-1), new_det_scores[i], self.track_num + i, new_det_features[i].view(1,-1),
																	self.inactive_patience, self.max_features_num))
		self.track_num += num_new

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		pos = self.get_pos()

		# regress
		_, scores, bbox_pred, rois = self.obj_detect.test_rois(pos)
		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
		pos = boxes[:, self.cl*4:(self.cl+1)*4]
		scores = scores[:, self.cl]

		s = []
		for i in range(len(self.tracks)-1,-1,-1):
			t = self.tracks[i]
			t.score = scores[i]
			if scores[i] <= self.regression_person_thresh:
				self.tracks_to_inactive([t])
			else:
				s.append(scores[i])
				# t.prev_pos = t.pos
				t.pos = pos[i].view(1,-1)
		return torch.Tensor(s[::-1]).cuda()

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks],0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def reid(self, blob, new_det_pos, new_det_scores):
		"""Tries to ReID inactive tracks with provided detections."""
		new_det_features = self.reid_network.test_rois(blob['app_data'][0], new_det_pos / blob['im_info'][0][2]).data
		if len(self.inactive_tracks) >= 1 and self.do_reid:
			# calculate appearance distances
			dist_mat = []
			pos = []
			for t in self.inactive_tracks:
				dist_mat.append(torch.cat([t.test_features(feat.view(1,-1)) for feat in new_det_features], 1))
				pos.append(t.pos)
			if len(dist_mat) > 1:
				dist_mat = torch.cat(dist_mat, 0)
				pos = torch.cat(pos,0)
			else:
				dist_mat = dist_mat[0]
				pos = pos[0]

			# calculate IoU distances
			iou = bbox_overlaps(pos, new_det_pos)
			iou_mask = torch.ge(iou, self.reid_iou_threshold)
			iou_neg_mask = ~iou_mask
			# make all impossible assignemnts to the same add big value
			dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
			dist_mat = dist_mat.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if dist_mat[r,c] <= self.reid_sim_threshold:
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_inactive = 0
					t.last_v = torch.Tensor([])
					t.pos = new_det_pos[c].view(1,-1)
					t.add_features(new_det_features[c].view(1,-1))
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
				new_det_scores = new_det_scores[keep]
				new_det_features = new_det_features[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
				new_det_scores = torch.zeros(0).cuda()
				new_det_features = torch.zeros(0).cuda()
		return new_det_pos, new_det_scores, new_det_features

	def clear_inactive(self):
		"""Checks if inactive tracks should be removed."""
		to_remove = []
		for t in self.inactive_tracks:
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		new_features = self.reid_network.test_rois(blob['app_data'][0], self.get_pos() / blob['im_info'][0][2]).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t,f in zip(self.tracks, new_features):
			t.add_features(f.view(1,-1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy()
			im2 = blob['data'][0][0].cpu().numpy()
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = self.warp_mode
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			number_of_iterations = self.number_of_iterations
			termination_eps = self.termination_eps
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for t in self.tracks:
				p = t.pos[0]
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)
				pos = torch.cat((p1_n, p2_n), 1).cuda()

				t.pos = pos.view(1,-1)
				#t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					p = t.pos[0]
					p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
					p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
					p1_n = torch.mm(warp_matrix, p1).view(1,2)
					p2_n = torch.mm(warp_matrix, p2).view(1,2)
					pos = torch.cat((p1_n, p2_n), 1).cuda()
					t.pos = pos.view(1,-1)

			if self.motion_model:
				for t in self.tracks:
					if t.last_pos.nelement() > 0:
						p = t.last_pos[0]
						p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
						p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

						p1_n = torch.mm(warp_matrix, p1).view(1,2)
						p2_n = torch.mm(warp_matrix, p2).view(1,2)
						pos = torch.cat((p1_n, p2_n), 1).cuda()

						t.last_pos = pos.view(1,-1)

	def motion(self):
		"""Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
		for t in self.tracks:
			# last_pos = t.pos.clone()
			# t.last_pos = last_pos
			# if t.last_pos.nelement() > 0:
				# extract center coordinates of last pos

			x1l = t.last_pos[0,0]
			y1l = t.last_pos[0,1]
			x2l = t.last_pos[0,2]
			y2l = t.last_pos[0,3]
			cxl = (x2l + x1l)/2
			cyl = (y2l + y1l)/2

			# extract coordinates of current pos
			x1p = t.pos[0,0]
			y1p = t.pos[0,1]
			x2p = t.pos[0,2]
			y2p = t.pos[0,3]
			cxp = (x2p + x1p)/2
			cyp = (y2p + y1p)/2
			wp = x2p - x1p
			hp = y2p - y1p

			# v = cp - cl, x_new = v + cp = 2cp - cl
			cxp_new = 2*cxp - cxl
			cyp_new = 2*cyp - cyl

			t.pos[0,0] = cxp_new - wp/2
			t.pos[0,1] = cyp_new - hp/2
			t.pos[0,2] = cxp_new + wp/2
			t.pos[0,3] = cyp_new + hp/2

			t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					# extract coordinates of current pos
					x1p = t.pos[0, 0]
					y1p = t.pos[0, 1]
					x2p = t.pos[0, 2]
					y2p = t.pos[0, 3]
					cxp = (x2p + x1p)/2
					cyp = (y2p + y1p)/2
					wp = x2p - x1p
					hp = y2p - y1p

					cxp_new = cxp + t.last_v[0]
					cyp_new = cyp + t.last_v[1]

					t.pos[0,0] = cxp_new - wp/2
					t.pos[0,1] = cyp_new - hp/2
					t.pos[0,2] = cxp_new + wp/2
					t.pos[0,3] = cyp_new + hp/2

	def step(self, blob):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		self.image = blob['image']
		# print(blob['image'].shape)
		self.video_path = blob['im_path']
		for t in self.tracks:
			t.last_pos = t.pos.clone()

		###########################
		# Look for new detections #
		###########################
		self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])
		if self.public_detections:
			dets = blob['dets']
			if len(dets) > 0:
				dets = torch.cat(dets, 0)[:, :4]
				_, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()
		else:
			_, scores, bbox_pred, rois = self.obj_detect.detect()

		if rois.nelement() > 0:
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

			# Filter out tracks that have too low person score
			scores = scores[:, self.cl]
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			boxes = boxes[inds]
			det_pos = boxes[:, self.cl*4:(self.cl+1)*4]
			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #
		##################
		num_tracks = 0
		nms_inp_reg = torch.zeros(0).cuda()
		if len(self.tracks):
			# align
			if self.do_align:
				self.align(blob)
			# apply motion model
			if self.motion_model:
				self.motion()
			#regress
			person_scores = self.regress_tracks(blob)

			if len(self.tracks):

				# create nms input
				# new_features = self.get_appearances(blob)

				# nms here if tracks overlap
				nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1, 1)), 1)
				keep = nms(nms_inp_reg, self.regression_nms_thresh)

				self.tracks_to_inactive([self.tracks[i]
				                         for i in list(range(len(self.tracks)))
				                         if i not in keep])

				if keep.nelement() > 0:
					nms_inp_reg = torch.cat((self.get_pos(), torch.ones(self.get_pos().size(0)).add_(3).view(-1,1).cuda()),1)
					new_features = self.get_appearances(blob)

					self.add_features(new_features)
					num_tracks = nms_inp_reg.size(0)
				else:
					nms_inp_reg = torch.zeros(0).cuda()
					num_tracks = 0

		#####################
		# Create new tracks #
		#####################

		# !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
		if det_pos.nelement() > 0:
			nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		else:
			nms_inp_det = torch.zeros(0).cuda()
		if nms_inp_det.nelement() > 0:
			keep = nms(nms_inp_det, self.detection_nms_thresh)
			nms_inp_det = nms_inp_det[keep]
			# check with every track in a single run (problem if tracks delete each other)
			for i in range(num_tracks):
				nms_inp = torch.cat((nms_inp_reg[i].view(1,-1), nms_inp_det), 0)
				keep = nms(nms_inp, self.detection_nms_thresh)
				keep = keep[torch.ge(keep,1)]
				if keep.nelement() == 0:
					nms_inp_det = nms_inp_det.new(0)
					break
				nms_inp_det = nms_inp[keep]

		if nms_inp_det.nelement() > 0:
			new_det_pos = nms_inp_det[:,:4]
			new_det_scores = nms_inp_det[:,4]

			# try to redientify tracks
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			pos = t.pos[0] / blob['im_info'][0][2]
			sc = t.score
			self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

		self.im_index += 1
		self.last_image = blob['data'][0][0]

		self.clear_inactive()

	def get_results(self):
		return self.results

	def draw_bboxes(self, img, bbox, identities=None, offset=(0, 0)):
		objects = []
		for i, box in enumerate(bbox):
			x1, y1, x2, y2 = [int(i) for i in box]
			x1 += offset[0]
			x2 += offset[0]
			y1 += offset[1]
			y2 += offset[1]

			# box text and bar
			id = int(identities[i]) if identities is not None else 0
			color = COLORS_10[id % len(COLORS_10)]
			# label = '{} {}'.format("id-", id)
			label = '{}'.format(id)
			img_crop = img[y1:y2, x1:x2]
			objects.append(img_crop)
			t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
			cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
			cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
		return img, objects

	def show_tracks(self, area):

		# self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
		# self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# self.area = 0, 0, self.im_width, self.im_height
		ori_im = self.image
		xmin, ymin, xmax, ymax = area

		# self.write_video = True
		# if self.write_video:
		# 	 fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		# 	 cv2.VideoWriter("demo1.avi", fourcc, 20, (self.im_width, self.im_height))

		bbox_output = []
		for t in self.tracks:
			t_id = int(t.id)
			# bbox_xyxy = t.pos
			x1 = int(t.pos[0, 0])
			y1 = int(t.pos[0, 1])
			x2 = int(t.pos[0, 2])
			y2 = int(t.pos[0, 3])

			# threshold = 30
			# if (x2 - x1 > threshold or y2 - y1 > threshold):
			bbox_output.append(np.array([x1, y1, x2, y2, t_id], dtype=np.int))
		# print("====bbox:", len(bbox_output))
		if len(bbox_output) > 0:
			bbox_output = np.stack(bbox_output, axis=0)
			bbox_xyxy = bbox_output[:, :4]
			identities = bbox_output[:, -1]
			ori_im, objects = self.draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))
			head_im, heads = self.step_head(ori_im, objects, bbox_output)
		
		cv2.imshow("output", head_im)
		cv2.waitKey(1)

		return objects, identities
		# if self.write_video:
		# 	self.output.write(ori_im)

	def step_head(self, ori_im, objects, bbox_output):

		for i in range(0, len(objects)):
			# input person objects to detect each head
			blobs_head, im_scales = test._get_blobs(objects[i])
			data = blobs_head['data']
			head = {}
			head['image'] = cv2.resize(objects[i], (0, 0), fx=im_scales, fy=im_scales, interpolation=cv2.INTER_NEAREST)
			head['data'] = torch.from_numpy(data).unsqueeze(0)
			im_info = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
			head['im_info'] = torch.from_numpy(im_info).unsqueeze(0)
			# convert to siamese input
			objects_cv = cv2.cvtColor(objects[i], cv2.COLOR_BGR2RGB)
			objects_cv = Image.fromarray(objects_cv)
			objects_cv = transforms(objects_cv)
			head['app_data'] = objects_cv.unsqueeze(0).unsqueeze(0)
			###############################
			# detect the head from object #
			###############################
			self.obj_detect_head.load_image(head['data'][0], head['im_info'][0])
			_, scores, bbox_pred, rois = self.obj_detect_head.detect()
			# filter the bbox by scores
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), head['im_info'][0][:2]).data
			scores = scores[:, self.cl]
			inds = torch.gt(scores, 0.1).nonzero().view(-1)
			if inds.nelement() > 0:
				boxes = boxes[inds]
				boxes = boxes / im_scales[0]
				det_pos = boxes[:, self.cl*4:(self.cl+1)*4]
				det_pos = det_pos[0, :4].view(1, 4)  # remain only one bbox
				object_pos = torch.from_numpy(bbox_output[i, :2]).repeat(1,2).float().cuda()
				det_pos += object_pos
				det_scores = scores[inds[0]]
			else:
				det_pos = torch.zeros(0).cuda()
				det_scores = torch.zeros(0).cuda()
			# print("======scores, pos:", det_scores, det_pos)
			# head_im, heads = self.draw_bboxes(objects[i], det_pos)
			i_id = [bbox_output[:, -1][i],]
			head_im, heads = self.draw_bboxes(ori_im, det_pos, i_id)
		
		return head_im, heads


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = torch.Tensor([])
		self.last_v = torch.Tensor([])
		self.gt_id = None

	def is_to_purge(self):
		"""Tests if the object has been too long inactive and is to remove."""
		self.count_inactive += 1
		self.last_pos = torch.Tensor([])
		if self.count_inactive > self.inactive_patience:
			return True
		else:
			return False

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(self.features, 0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features)
		return dist
