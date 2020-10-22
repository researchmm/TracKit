import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from os.path import join, exists
from utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round, get_subwindow_tracking_mask


class OceanPlus(object):
    def __init__(self, info):
        super(OceanPlus, self).__init__()
        self.info = info   # model and benchmark info
        self.stride = 8

        if info.dataset in ['DAVIS2016', 'DAVIS2017', 'YTBVOS']:
            self.vos = True
        else:
            self.vos = False

    def init(self, im, target_pos, target_sz, model, hp=None, online=False, mask=None, debug=False):
        # in: whether input infrared image
        state = dict()
        # epoch test
        p = AdaConfig()

        self.debug = debug

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        self.imh = state['im_h']
        self.imw = state['im_w']

        # single test
        # if not hp and not self.info.epoch_test:
        if True:
            prefix = [x for x in ['OTB', 'VOT', 'DAVIS'] if x in self.info.dataset]
            if len(prefix) == 0: prefix = [self.info.dataset]
            absPath = os.path.abspath(os.path.dirname(__file__))
            yname='OceanPlus.yaml'
            yamlPath = os.path.join(absPath, '../../experiments/test/{}/'.format(prefix[0]), yname)
            cfg = load_yaml(yamlPath)
         
            if self.info.dataset not in list(cfg.keys()):
                print('[*] unsupported benchmark, use VOT2020 hyper-parameters (not optimal)')
                cfg_benchmark = cfg['VOT2020']
            else:
                cfg_benchmark = cfg[self.info.dataset]
      
            p.update(cfg_benchmark)
            p.renew()

            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = cfg_benchmark['big_sz']
                p.renew()
            else:
                p.instance_size = cfg_benchmark['small_sz']
                p.renew()

        self.grids(p)   # self.grid_to_search_x, self.grid_to_search_y

        net = model
        # param tune
        if hp:
            p.update(hp)
            if 'lambda_u' in hp.keys() or 'lambda_s' in hp.keys():
                net.update_lambda(hp['lambda_u'], hp['lambda_s'])
            if 'iter1' in hp.keys() or 'iter2' in hp.keys():
                 net.update_iter(hp['iter1'], hp['iter2'])

            print('======= hyper-parameters: pk: {:.3f}, wi: {:.2f}, lr: {:.2f} ======='.format(p.penalty_k, p.window_influence, p.lr))
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        mask_crop, _ = get_subwindow_tracking_mask(mask, target_pos, p.exemplar_size, s_z, out_mode=None)
        mask_crop = (mask_crop > 0.5).astype(np.uint8)
        mask_crop = torch.from_numpy(mask_crop)

        # vis zcrop
        # vis = 0.5 * z_crop.permute(1,2,0) + 255 *  mask_crop.unsqueeze(-1).float()
        # cv2.imwrite('zcrop.jpg', vis.numpy())

        z = z_crop.unsqueeze(0)
        net.template(z.cuda(), mask_crop.unsqueeze(0).cuda())


        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        self.p = p
        self.debug_on_crop = False
        self.debug_on_ori = False
        self.save_mask = False  # save all mask results
        self.mask_ratio = False

        self.update_template = True

        if self.debug_on_ori or self.debug_on_crop:
            print('Warning: debuging...')
            print('Warning: turning off debugging mode after this process')
            self.debug = True

        return state

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p):

        cls_score, bbox_pred, mask = net.track(x_crops)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2-pred_x1, pred_y2-pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2-pred_x1) / (pred_y2-pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        if self.online_score is not None:
            pscore_ori = pscore * (1 - p.window_influence) + window * p.window_influence
        else:
            pscore = pscore * (1 - p.window_influence) + window * p.window_influence
            pscore_ori = pscore

        if self.online_score is not None:
            s_size = pscore.shape[0]
            o_score = cv2.resize(self.online_score, (s_size, s_size), interpolation=cv2.INTER_CUBIC)
            pscore = p.online_ratio * o_score + (1 - p.online_ratio) * pscore_ori
        else:
            pass

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        if pscore_ori[r_max, c_max] > 0.95 and self.update_template:  # donot update for vos dataset
            pos_in_crop = np.array([diff_xs, diff_ys]) * scale_z   
            sz_in_crop = target_sz * scale_z
            net.update_roi_template(pos_in_crop, sz_in_crop, pscore[r_max, c_max])  # update template

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        if self.debug:
            bbox = [int(target_pos[0]-target_sz[0]/2), int(target_pos[1]-target_sz[1]/2), int(target_pos[0]+target_sz[0]/2), int(target_pos[1]+target_sz[1]/2)]

        # -----------------------  mask --------------------
        mask = mask.squeeze()
        mask = F.softmax(mask, dim=0)[1]
        mask = mask.squeeze().cpu().data.numpy()  # [255, 255]
        # print('---- in track0')
        if self.debug_on_crop:
            print('===========> debug on crop image <==========')
            # draw on crop image
            polygon = self.mask2box(mask, method='cv2poly')
            im = x_crops.squeeze().permute(1, 2, 0).cpu().data.numpy()
            output = self.draw_mask(mask, im, polygon=polygon, mask_ratio=0.8, draw_contour=False, object_num=1)
            cv2.imwrite('mask.jpg', output)
        else:
            # print('===========> debug on original image <==========')
            # width and height of original image patch in get_sub_window tracking
            context_xmin, context_xmax, context_ymin, context_ymax = self.crop_info['crop_cords']
            top_pad, left_pad, r, c = self.crop_info['pad_info']

            temp_w = context_xmax - context_xmin + 1
            temp_h = context_ymax - context_ymin + 1
            mask_temp = cv2.resize(mask, (int(temp_h), int(temp_w)), interpolation=cv2.INTER_CUBIC)

            # return mask to original image patch in get_sub_window tracking
            empty_mask = self.crop_info['empty_mask']
            empty_mask[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)] = mask_temp

            # remove crop padding
            mask_in_im = empty_mask[top_pad:top_pad + r, left_pad:left_pad + c]
            
            if self.debug_on_ori or self.debug:
                polygon = self.mask2box(mask_in_im, method='cv2poly')
                output = self.draw_mask(mask_in_im, self.im_ori, polygon=polygon, box=bbox, mask_ratio=0.8, draw_contour=False, object_num=1)
                cv2.imwrite(join(self.save_dir, self.name.split('/')[-1]), output)
            else:
                polygon = None

            # ------ test -------
            results = dict()
            results['target_pos'] = target_pos
            results['target_sz'] = target_sz
            results['cls_score'] = cls_score[r_max, c_max]
            results['mask'] = (mask_in_im > self.p.seg_thr).astype(np.uint8)
            results['mask_ori'] = mask_in_im
            results['polygon'] = polygon

        return results

    def track(self, state, im, online_score=None, gt=None, name=None):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']
        self.im_ori = im.copy()
        self.gt = gt

        if online_score is not None:
            self.online_score = online_score.squeeze().cpu().data.numpy()
        else:
            self.online_score = None

        # debug
        if self.debug:
            temp = name.split('/')[-2]
            self.name = name
            self.save_dir = join('debug', temp)
            if not exists(self.save_dir):
                os.makedirs(self.save_dir)

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        x_crop, self.crop_info = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        x_crop = x_crop.unsqueeze(0)

        results = self.update(net, x_crop.cuda(), target_pos, target_sz*scale_z, window, scale_z, p)

        target_pos, target_sz, cls_score, mask, mask_ori, polygon = results['target_pos'], results['target_sz'], results['cls_score'], results['mask'], results['mask_ori'], results['polygon']
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['cls_score'] = cls_score
        state['mask'] = mask
        state['mask_ori'] = mask_ori
        state['polygon'] = polygon
        state['p'] = p

        return state

    def mask2box(self, mask, method='cv2poly'):
        """
        method: cv2poly --> opencv
                opt --> vot version
        """
        mask = (mask > self.p.seg_thr).astype(np.uint8)
        if method == 'cv2poly':
            if cv2.__version__[-5] == '4':
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            else:
                _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cnt_area = [cv2.contourArea(cnt) for cnt in contours]
            if len(contours) != 0 and np.max(cnt_area) > 0:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
                # box_in_img = pbox
                prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
                pred_polygon = ((prbox[0][0], prbox[0][1]), (prbox[1][0], prbox[1][1]),
                                (prbox[2][0], prbox[2][1]), (prbox[3][0], prbox[3][1]))

                return pred_polygon
            else:
                return None

        elif method == 'opt':
            pass
        else:
            raise ValueError('not supported mask2box methods')

    def draw_mask(self, mask, im, polygon=None, box=None, mask_ratio=0.2, draw_contour=False, object_num=1):
        # draw mask
        # mask: 0, 255
        mask = mask > self.p.seg_thr
        mask = mask.astype('uint8')
        # COLOR
        COLORS = np.random.randint(128, 255, size=(object_num, 3), dtype="uint8")
        COLORSIM = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask_draw = COLORSIM[mask]

        # mask = mask * 255

        where_is = (mask == 0).astype(int)
        where_is = np.expand_dims(where_is, axis=-1)
        out_mask = where_is * im
        output = ((1 - mask_ratio) * im + mask_ratio * mask_draw + mask_ratio * out_mask).astype("uint8")
        # output = ((1 - mask_ratio) * im + mask_ratio * mask).astype("uint8")

        if draw_contour:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            try:
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # remove small contours
                areas = np.array([cv2.contourArea(c) for c in contours])
                max_area = np.max(areas)
                max_idx = np.argmax(areas)

                minArea = max_area * 0.01

                filteredContours = []
                findhier = []
                for id, i in enumerate(contours):
                    area = cv2.contourArea(i)
                    if area > minArea:
                        filteredContours.append(i)
                        findhier.append(hierarchy[:, id, :])

                # findhier = np.array(findhier).transpose(1, 0, 2)
                output = cv2.drawContours(output, filteredContours, -1, (255, 255, 255), 2, cv2.LINE_8)
            except:
                print('draw contour process fails...')
        else:
            pass

        if polygon is not None:
            # draw rotated box
            polygon = np.int0(polygon)  # to int
            output = cv2.polylines(output, [polygon.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            # output = cv2.drawContours(output, [polygon], 0, (0, 0, 255), 3)

        # draw gt
        try:
            gt = ((self.gt[0], self.gt[1]), (self.gt[2], self.gt[3]), (self.gt[4], self.gt[5]), (self.gt[6], self.gt[7]))
            gt = np.int0(gt)  # to int
            output = cv2.polylines(output, [gt.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
        except:
            pass

        if box is not None:
            output = cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

        return output

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2


    def IOUgroup(self, pred_x1, pred_y1, pred_x2, pred_y2, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy

        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class AdaConfig(object):
    penalty_k = 0.06
    window_influence = 0.484
    lr = 0.644
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
    context_amount = 0.5
    ratio = 0.94
    online_ratio = 0.7
    #seg_thr = 0.84
    seg_thr = 0.9
    lambda_u = 0.1
    lambda_s = 0.2
    iter1 = 0.33
    iter2 = 0.33


    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
