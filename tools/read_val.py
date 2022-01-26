# import pickle
# import pprint
#
# file = open(
# 	"/home/nrsl/workspace/git/detection3d/OpenPCDet/output/kitti_models/pv_rcnn/default/eval/epoch_8369/test/default/result.pkl",
# 	"rb")
# data = pickle.load(file)
# pprint.pprint(data)
# file.close()

import easydict

cfg = easydict.EasyDict()
cfg['123'] = 1
print(cfg)
cfg.update(easydict.EasyDict({'123': 2}))
print(cfg)
