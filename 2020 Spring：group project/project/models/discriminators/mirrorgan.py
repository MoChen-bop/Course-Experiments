import sys
sys.path.append('../..')

import torch


def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
    	conv3x3(in_planes, out_planes),
    	nn.BatchNorm2d(out_planes),
    	nn.LeakyReLU(0.2, inplace=True))
    return block


def downBlock(in_planes, out_planes):
    block = nn.Sequential(
    	nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
    	nn.BatchNorm2d(out_planes),
    	nn.LeakyReLU(0.2, inplace=True))
    return block


def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
    	nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
    	nn.LeakyReLU(0.2, inplace=True),

    	nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
    	nn.BatchNorm2d(ndf * 2),
    	nn.LeakyReLU(0.2, inplace=True),

    	nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
    	nn.BatchNorm2d(ndf * 4),
    	nn.LeakyReLU(0.2, inplace=True),

    	nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
    	nn.BatchNorm2d(ndf * 8),
    	nn.LeakyReLU(0.2, inplace=True))
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
        	nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
        	nn.Sigmoid())


    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)

            h_c_code = torch.cat((h_code, c_code), 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        ndf = cfg.MODEL.MIRRORGAN.DF_DIM 
        nef = cfg.MODEL.MIRRORGAN.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)


    def forward(self, x_var):
        x_code64 = self.img_code_s16(x_var)
        return x_code64


# need to continue 