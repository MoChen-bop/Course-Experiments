import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.parallel


class icml_G(nn.Module):
    
    def __init__(self, nc, ngf, deploy=False):
        super(icml_G, self).__init__()
        self.linear = nn.Linear(1024, 128)
        self.convt_228_512 = nn.ConvTranspose2d(100 + 128, ngf * 8, 4, 1, 0, bias=False)

        self.conv_512_128 = nn.Conv2d(ngf * 8, ngf  * 2, 1, 1, 0, bias=False)
        self.conv_128_128 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False)
        self.conv_128_512 = nn.Conv2d(ngf * 2, ngf * 8, 3, 1, 1, bias=False)

        self.convt_512_256 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)

        self.conv_256_64 = nn.Conv2d(ngf * 4, ngf, 1, 1, 0, bias=False)

        self.conv_64_64  = nn.Conv2d(ngf,     ngf, 3, 1, 1, bias=False)
        self.conv_64_256 = nn.Conv2d(ngf, ngf * 4, 3, 1, 1, bias=False)

        self.convt_256_128 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.convt_128_64 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.convt_64_3   = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.bn_512 = nn.BatchNorm2d(ngf * 8)
        self.bn_256 = nn.BatchNorm2d(ngf * 4)
        self.bn_128 = nn.BatchNorm2d(ngf * 2)
        self.bn_64  = nn.BatchNorm2d(ngf)

        self.deploy = deploy



    def forward(self, txt, noise):
        batch_size = txt.size(0)

        txt_emb = F.leaky_relu(self.linear(txt.view(batch_size, 1024)), 0.2, True) # batch_size x 128
        noise = noise.reshape(batch_size, -1, 1, 1)
        input = torch.cat((txt_emb.view(batch_size, 128, 1, 1), noise), 1) # batch_size x (128 + 100) x 1 x 1

        e1 = self.bn_512(self.convt_228_512(input)) # batch_size x 512 x 4 x 4

        e2 = F.relu(self.bn_128(self.conv_512_128(e1)), True) # batch_size x 128 x 4 x 4
        e3 = F.relu(self.bn_128(self.conv_128_128(e2)), True) # batch_size x 128 x 4 x 4
        e4 = self.bn_512(self.conv_128_512(e3))               # batch_size x 512 x 4 x 4

        e5 = self.bn_256(self.convt_512_256(F.relu((e1 + e4), True))) # batch_size x 256 x 8 x 8
        
        e6 = F.relu(self.bn_64(self.conv_256_64(e5)), True) # batch_size x  64 x 8 x 8
        e7 = F.relu(self.bn_64(self.conv_64_64(e6)), True)  # batch_size x  64 x 8 x 8
        e8 = self.bn_256(self.conv_64_256(e7))              # batch_size x 256 x 8 x 8

        e9 = F.relu(self.bn_128(self.convt_256_128(F.relu((e5 + e8), True)))) # batch_size x 128 x 16 x 16
        e10 = F.relu(self.bn_64(self.convt_128_64(e9))) # batch_size x 64 x 32 x 32
        o = F.tanh(self.convt_64_3(e10)) # batch_size x 3 x 64 x 64

        return o


if __name__ == '__main__':
    
    txt = torch.rand((4, 1024))
    noise = torch.rand((4, 100))
    model_G = icml_G(3, 64)
    o = model_G(txt, noise)
    print(o.shape)