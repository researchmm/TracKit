import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .connect import xcorr_depthwise

class ARN(nn.Module):
    """
    Attention Retrieval Network in Ocean+
    """
    def __init__(self, inchannels=256, outchannels=256):
        super(ARN, self).__init__()
        self.s_embed = nn.Conv2d(inchannels, outchannels, 1)  # embedding for search feature
        self.t_embed = nn.Conv2d(inchannels, outchannels, 1)  # embeeding for template feature

    def forward(self, xf, zf, zf_mask):
        # xf: [B, C, H, W]
        # zf: [B, C, H, W]
        # zf_mask: [B, H, W]
        # pdb.set_trace()
        xf = self.s_embed(xf)
        zf = self.t_embed(zf)

        B, C, Hx, Wx = xf.size()
        B, C, Hz, Wz = zf.size()

        xf = xf.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        xf = xf.view(B, -1, C)   # [B, H*W, C]
        zf = zf.view(B, C, -1)   # [B, C, H*W]

        att = torch.matmul(xf, zf)  # [HW, HW]
        att = att / math.sqrt(C)
        att = F.softmax(att, dim=-1)  # [HW, HW]
        zf_mask = nn.Upsample(size=(Hz, Wz), mode='bilinear', align_corners=True)(zf_mask.unsqueeze(1))
        # zf_mask = (zf_mask > 0.5).float()
        zf_mask = zf_mask.view(B, -1, 1)

        arn = torch.matmul(att, zf_mask)  # [B, H*W]
        arn = arn.view(B, Hx, Hx).unsqueeze(1)
        return arn

class MSS(nn.Module):
    """
    Multi-resolution Single-stage Segmentation (fast, used for VOT-RT)
    """

    def __init__(self):
        super(MSS, self).__init__()
        # BACKBONE
        self.b4 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(512, 32, 3, padding=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(256, 16, 3, padding=1), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(64, 4, 3, padding=1), nn.ReLU())
        self.b0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1), nn.ReLU())

        # REFINE
        self.rCo = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU())
        self.r3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.r2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.r1 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.r0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))

        # multi refine layer
        self.m3 = nn.Sequential(nn.Conv2d(32, 2, 3, padding=1))  # stride = 8
        self.m2 = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1))  # stride = 4
        self.m1 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))   # stride = 2

        self.multi_loss = True

        # for sequential
        self.sequential = ARN(256, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, features, input_size=None, zf_ori=None, template_mask=None):
        b1, b2, b3, b4, corr = features

        b4_size = b4.size()[2:]
        b3_size = b3.size()[2:]
        b2_size = b2.size()[2:]
        b1_size = b1.size()[2:]
        if input_size is None: input_size = (255, 255)

        # prepare for sequential model
        arn = self.sequential(b4, zf_ori, template_mask)  # [B, H, W]
        arn = torch.clamp(arn, 0, 1)
        b4 = b4 + arn

        corr = nn.Upsample(size=b4_size, mode='bilinear', align_corners=True)(corr)
        r4 = self.rCo(corr) + self.b4(b4)

        r4 = nn.Upsample(size=b3_size, mode='bilinear', align_corners=True)(r4)
        r3 = self.r3(r4) + self.b3(b3)

        # r3up + b2
        r3 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(r3)
        r2 = self.r2(r3) + self.b2(b2)


        # r2up + b1
        r2 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(r2)
        r1 = self.r1(r2) + self.b1(b1)

        # r1 up
        r1 = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(r1)
        mask = self.r0(r1)

        return mask

class MMS(nn.Module):
    def __init__(self):
        """
        Multi-resolution Multi-stage Segmentation (suitable for VOS)
        """
        super(MMS, self).__init__()
        # BACKBONE
        self.b4 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(512, 32, 3, padding=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(256, 16, 3, padding=1), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(64, 4, 3, padding=1), nn.ReLU())
        self.b0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1), nn.ReLU())

        # REFINE
        self.rCo = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU())
        self.r3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.r2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.r1 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.r0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))

        # being used in training (not inference)
        self.m3 = nn.Sequential(nn.Conv2d(32, 2, 3, padding=1))  # stride = 8
        self.m2 = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1))  # stride = 4
        self.m1 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))   # stride = 2

        # for sequential
        self.sequential = ARN(256, 64)   # transduction attention
        self.iter = IterRefine()
        self.iter2 = IterRefine2()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
         
        self.ratio1, self.ratio2, self.ratio3 = 0.33, 0.33, 0.33 
    def forward(self, features, input_size=None, zf_ori=None, template_mask=None):
        b1, b2, b3, b4, corr = features

        b4_size = b4.size()[2:]
        b3_size = b3.size()[2:]
        b2_size = b2.size()[2:]
        b1_size = b1.size()[2:]
        if input_size is None: input_size = (255, 255)

        # iter list -- return for iter list
        iterList = []

        # transduction network
        arn = self.sequential(b4, zf_ori, template_mask)  # [B, H, W]
        arn = torch.clamp(arn, 0, 1)
        b4 = b4 + arn
     
        corr = nn.Upsample(size=b4_size, mode='bilinear', align_corners=True)(corr)
        corr = self.rCo(corr)
        b4 = self.b4(b4)
        r4 = corr + b4
        iterList.append(r4)  # [64]

        b3 = self.b3(b3)
        iterList.append(b3)  # [64, 32]

        r3 = self.r3(r4) + b3
        r3 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(r3)

        b2 = self.b2(b2)   # [64, 32, 16]
        iterList.append(b2)  # [64, 32, 16]
        r2 = self.r2(r3) + b2
        r2 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(r2)

        b1 = self.b1(b1)   # [64, 32, 16, 4]
        iterList.append(b1)
        r1 = self.r1(r2) + b1

        # r1 up
        r1 = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(r1)
        mask = self.r0(r1)

        ##### iter refine
        mask_iter, flist = self.iter(iterList, pre_mask = mask, input_size=input_size)
        mask_list = self.iter2(flist, pre_mask=mask_iter, input_size=input_size)
        
        return self.ratio1 * mask + self.ratio2 * mask_iter[0] + self.ratio3 * mask_list

    def update_iter(self, ratio1, ratio2, ratio3):
        self.ratio1, self.ratio2, self.ratio3 = ratio1, ratio2, ratio3


class IterRefine(nn.Module):
    def __init__(self):
        """
        stage2 of MMS 
        TODO: simplify the code
        """
        super(IterRefine, self).__init__()
        # BACKBONE
        self.b3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        # REFINE
        self.r4 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.r3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.r2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.r1 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.r0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))

        # being used in training (not inference)
        self.m3 = nn.Sequential(nn.Conv2d(32, 2, 3, padding=1))  # stride = 8
        self.m2 = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1))  # stride = 4
        self.m1 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))   # stride = 2

        self.sequential = ARN(256, 64)   # transduction attention

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, fList, pre_mask=None, input_size=None):
        b4, b3, b2, b1 = fList   # [64, 32, 16, 4]
        iterList = []

        # pre_mask processing
        att = F.softmax(pre_mask, dim=1)[:,1,...].unsqueeze(1)
        att = torch.clamp(att, 0.7, 1)

        b4_size = b4.size()[2:]
        b3_size = b3.size()[2:]
        b2_size = b2.size()[2:]
        b1_size = b1.size()[2:]
        if input_size is None: input_size = (255, 255)

        att_b4 = nn.Upsample(size=b4_size, mode='bilinear', align_corners=True)(att)
        att_b2 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(att)
        att_b1 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(att)
        b4, b3, b2, b1 = att_b4 * b4, att_b4 * b3, att_b2 * b2, att_b1 * b1

        r4 = self.r4(b4)
        iterList.append(r4)

        b3 = self.b3(b3)
        iterList.append(b3)
        r3 = self.r3(r4) + b3   # 32

        # r3up + b2
        r3 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(r3)
        b2 = self.b2(b2)
        iterList.append(b2)
        r2 = self.r2(r3) +  b2 # 16

        # r2up + b1
        r2 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(r2)
        b1 = self.b1(b1)
        iterList.append(b1)
        r1 = self.r1(r2)  + b1


        # r1 up
        r1 = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(r1)
        mask = self.r0(r1)
        
        return [mask], iterList


class IterRefine2(nn.Module):
    def __init__(self):
        """
        stage3 of MMS 
        TODO: simplify the code
        """
        super(IterRefine2, self).__init__()
        # BACKBONE
        self.b3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        # REFINE
        self.r4 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.r3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.r2 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU())
        self.r1 = nn.Sequential(nn.Conv2d(8, 4, 3, padding=1), nn.ReLU())
        self.r0 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))

        # being used in training (not inference)
        self.m3 = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1))  # stride = 8
        self.m2 = nn.Sequential(nn.Conv2d(8, 2, 3, padding=1))  # stride = 4
        self.m1 = nn.Sequential(nn.Conv2d(4, 2, 3, padding=1))  # stride = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, fList, pre_mask=None, input_size=None):
        b4, b3, b2, b1 = fList  # [32, 32, 16, 4]
        iterList = []
        # pre_mask processing
        att = F.softmax(pre_mask[0], dim=1)[:, 1, ...].unsqueeze(1)
        att = torch.clamp(att, 0.7, 1)

        b4_size = b4.size()[2:]
        b3_size = b3.size()[2:]
        b2_size = b2.size()[2:]
        b1_size = b1.size()[2:]
        if input_size is None: input_size = (255, 255)

        att_b4 = nn.Upsample(size=b4_size, mode='bilinear', align_corners=True)(att)
        att_b2 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(att)
        att_b1 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(att)
        b4, b3, b2, b1 = att_b4 * b4, att_b4 * b3, att_b2 * b2, att_b1 * b1

        r4 = self.r4(b4)
        b3 = self.b3(b3)
        r3 = self.r3(r4) + b3  # 16

        # r3up + b2
        r3 = nn.Upsample(size=b2_size, mode='bilinear', align_corners=True)(r3)
        b2 = self.b2(b2)
        r2 = self.r2(r3) + b2  # 8

        # r2up + b1
        r2 = nn.Upsample(size=b1_size, mode='bilinear', align_corners=True)(r2)
        b1 = self.b1(b1)
        r1 = self.r1(r2) + b1

        # r1 up
        r1 = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(r1)
        mask = self.r0(r1)

        return mask



