import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.parallel


class icml_D(nn.Module):
    
    def __init__(self, nc, ndf):
        super(icml_D, self).__init__()

        self.linear = nn.Linear(1024, 128)

        self.conv_3_64 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv_64_128 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv_128_256 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv_256_512 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)

        self.conv_512_128 = nn.Conv2d(ndf * 8, ndf * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ndf * 2, ndf * 8, 3, 1, 1, bias=False)

        self.conv_640_512 = nn.Conv2d(ndf * 8 + 128, ndf * 8, 1, 1, 0, bias=False)
        self.conv_512_1 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.bn_128 = nn.BatchNorm2d(ndf * 2)
        self.bn_256 = nn.BatchNorm2d(ndf * 4)
        self.bn_512 = nn.BatchNorm2d(ndf * 8)


    def forward(self, img, txt):
        batch_size = img.size(0)

        e1 = F.leaky_relu(self.conv_3_64(img), 0.2, True) # batch_size x 64 x 32 x 32
        e2 = F.leaky_relu(self.bn_128(self.conv_64_128(e1)), 0.2, True) # batch_size x 128 x 16 x 16
        e3 = F.leaky_relu(self.bn_256(self.conv_128_256(e2)), 0.2, True) # batch_size x 256 x 8 x 8
        e4 = self.bn_512(self.conv_256_512(e3)) # batch_size x 512 x 4 x 4

        e5 = F.leaky_relu(self.bn_128(self.conv_512_128(e4)), 0.2, True) # batch_size x 128 x 4 x 4
        e6 = F.leaky_relu(self.bn_128(self.conv_128_128(e5)), 0.2, True) # batch_size x 128 x 4 x 4
        e7 = self.bn_512(self.conv_128_512(e6)) # batch_size x 512 x 4 x 4

        e8 = F.leaky_relu((e4 + e7), 0.2, True) # batch_size x 512 x 4 x 4

        txt_emb = F.leaky_relu(self.linear(txt.view(batch_size, 1024)), 0.2, True) # batch_size x 128
        t = txt_emb.view(batch_size, 128, 1, 1) # batch_size x 128 x 1 x 1
        e9 = t.expand(batch_size, 128, 4, 4)    # batch_size x 128 x 4 x 4

        e12 = torch.cat((e8, e9), 1) # batch_size x (128 + 512) x 4 x 4
        e13 = F.leaky_relu(self.bn_512(self.conv_640_512(e12)), 0.2, True) # batcj_size x 512 x 4 x 4
        e14 = self.conv_512_1(e13) # batch_size x 1 x 1 x 1
        o = e14.view(batch_size)
        o = F.sigmoid(o)

        return o


if __name__ == '__main__':
    
    img = torch.randn((4, 3, 64, 64))
    txt = torch.randn((4, 1024))
    model_d = icml_D(3, 64)
    o = model_d(img, txt)
    print(o.shape)