import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class OceanPlus_(nn.Module):
    def __init__(self):
        super(OceanPlus_, self).__init__()
        self.features = None
        self.connect_model = None
        self.mask_model = None
        self.zf = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.neck = None
        self.search_size = 255
        self.score_size = 25
        self.batch = 32 if self.training else 1
        self.lambda_u = 0.1
        self.lambda_s = 0.2

        # self.grids()

    def feature_extractor(self, x, online=False):
        return self.features(x, online=online)

    def extract_for_online(self, x):
        xf = self.feature_extractor(x, online=True)
        return xf

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def update_roi_template(self, target_pos, target_sz, score):
        """
        :param target_pos:  pos in search (not the original)
        :param target_sz:  size in target size
        :param score:
        :return:
        """

        lambda_u = self.lambda_u * float(score)
        lambda_s = self.lambda_s
        N, C, H, W = self.search_size
        stride = 8
        assert N == 1, "not supported"
        l = W // 2
        x = range(-l, l + 1)
        y = range(-l, l + 1)

        hc_z = (target_sz[1] + 0.3 * sum(target_sz)) / stride
        wc_z = (target_sz[0] + 0.3 * sum(target_sz)) / stride
        grid_x = np.linspace(- wc_z / 2, wc_z / 2, 17)
        grid_y = np.linspace(- hc_z / 2, hc_z / 2, 17)
        grid_x = grid_x[5:-5] + target_pos[0] / stride
        grid_y = grid_y[5:-5] + target_pos[1] / stride
        x_offset = grid_x / l
        y_offset = grid_y / l

        grid = np.reshape(np.transpose([np.tile(x_offset, len(y_offset)), np.repeat(y_offset, len(x_offset))]), (len(grid_y), len(grid_x), 2))
        grid = torch.from_numpy(grid).unsqueeze(0).cuda()

        zmap = nn.functional.grid_sample(self.xf.double(), grid).float()
        # cls_kernel = self.rpn.cls.make_kernel(zmap)
        self.MA_kernel = (1 - lambda_u) * self.MA_kernel + lambda_u * zmap
        self.zf_update = self.zf * lambda_s + self.MA_kernel * (1.0 - lambda_s)

    def template(self, z, template_mask = None):
        _, self.zf = self.feature_extractor(z)

        if self.neck is not None:
            self.zf_ori, self.zf = self.neck(self.zf, crop=True)
    
        self.template_mask = template_mask.float()
        self.MA_kernel = self.zf.detach()
        self.zf_update = None
    

    def track(self, x):

        features_stages, xf = self.feature_extractor(x)

        if self.neck is not None:
            xf = self.neck(xf, crop=False)

        features_stages.append(xf)
        bbox_pred, cls_pred,  cls_feature, reg_feature = self.connect_model(xf, self.zf, update=self.zf_update)

        features_stages.append(cls_feature)
        pred_mask = self.mask_model(features_stages, input_size=x.size()[2:], zf_ori=self.zf_ori, template_mask=self.template_mask)
        self.search_size = xf.size()
        self.xf = xf.detach()
        
        return cls_pred, bbox_pred, pred_mask












