import torch

def my_transform(state_dict):
	state_dict_load = state_dict['model']
	# print("=====test", list(state_dict_load))
	
	for i in list(state_dict_load):
		if 'RCNN_base.0' in i:
			name = i.replace('RCNN_base.0', 'resnet.conv1')
		if 'RCNN_base.1' in i:
			name = i.replace('RCNN_base.1', 'resnet.bn1')
		if 'RCNN_base.4' in i:
			name = i.replace('RCNN_base.4', 'resnet.layer1')
		if 'RCNN_base.5' in i:
			name = i.replace('RCNN_base.5', 'resnet.layer2')
		if 'RCNN_base.6' in i:
			name = i.replace('RCNN_base.6', 'resnet.layer3')
		if 'RCNN_top.0' in i:
			name = i.replace('RCNN_top.0', 'resnet.layer4')
		if 'RCNN_rpn.RPN_Conv' in i:
			name = i.replace('RCNN_rpn.RPN_Conv', 'rpn_net')
		if 'RCNN_rpn.RPN_cls_score' in i:
			name = i.replace('RCNN_rpn.RPN_cls_score', 'rpn_cls_score_net')
		if 'RCNN_rpn.RPN_bbox_pred' in i:
			name = i.replace('RCNN_rpn.RPN_bbox_pred', 'rpn_bbox_pred_net')
		if 'RCNN_cls_score' in i:
			name = i.replace('RCNN_cls_score', 'cls_score_net')
		if 'RCNN_bbox_pred' in i:
			name = i.replace('RCNN_bbox_pred', 'bbox_pred_net')

		# print("===org:", state_dict_load[i])
		# t1 = state_dict_load[i]
		state_dict_load[name] = state_dict_load.pop(i)
		# t2 = state_dict_load[name]
		# print("=====test:", torch.equal(t1, t2))
		# print("===now:", state_dict_load[name])
	# print("=====test", list(state_dict_load))
	return state_dict_load