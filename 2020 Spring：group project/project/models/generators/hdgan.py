import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math
import functools


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def branch_out(in_dim, out_dim=3):
    _layers = [nn.ReflectionPad2d(1),
               nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=0, bias=False)]
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)


def pad_conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True, 
	use_bias=False, activation=nn.ReLU(True)):
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]

    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
        norm_layer(dim_out)]
    if use_activation:
        seq += [activation]

    return nn.Sequential(*seq)


class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()

        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear = nn.Linear(noise_dim, emb_dim * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def sample_encoded_context(self, mean, logsigma, kl_loss=False):
        epsilon = Variable(torch.cuda.FloatTensor(mean.size()).normal_())
        stddev = logsigma.exp()

        return epsilon.mul(stddev).add_(mean)


    def forward(self, inputs, kl_loss=True):
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma)
        return c, mean, log_sigma


class Sent2FeatMap(nn.Module):
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row * col * channel
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)


    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.ReLU(True)

        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ),
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]

        self.res_block = nn.Sequential(*seq)


    def forward(self, input):
        return self.res_block(input) + input


class Generator(nn.Module):
    
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, num_resblock=1, side_output_at=[64, 128, 256], deploy=False):
        super(Generator, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        act_layer = nn.ReLU(True)
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)

        self.vec_to_tensor = Sent2FeatMap(emb_dim + noise_dim, 4, 4, self.hid_dim * 8)
        self.side_output_at = side_output_at

        reduce_dim_at = [8, 32, 128, 256]
        num_scales = [4, 8, 16, 32, 64, 128, 256]

        cur_dim = self.hid_dim * 8
        for i in range(len(num_scales)):
            seq = []

            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]

            if num_scales[i] in reduce_dim_at:
                seq += [pad_conv_norm(cur_dim, cur_dim // 2, norm_layer, activation=act_layer)]
                cur_dim = cur_dim // 2

            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim)]

            setattr(self, 'scale_%d' % (num_scales[i]), nn.Sequential(*seq))

            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d' % (num_scales[i]), branch_out(cur_dim))
        self.apply(weight_init)

        self.deploy = deploy


    def forward(self, sent_embeddings, z):
        sent_random, mean, logsigma = self.condEmbedding(sent_embeddings)

        text = torch.cat([sent_random, z], dim=1)
        x = self.vec_to_tensor(text) # batch_size x (hid_dim * 8) x 4 x 4
        x_4 = self.scale_4(x) # batch_size x (hid_dim * 8) x 4 x 4
        x_8 = self.scale_8(x_4) # batch_size x (hid_dim * 4) x 8 x 8
        x_16 = self.scale_16(x_8) # batch_size x (hid_dim * 4) x 16 x 16
        x_32 = self.scale_32(x_16) # batch_size x (hid_dim * 2) x 32 x 32

        x_64 = self.scale_64(x_32) # batch_size x (hid_dim * 2) x 64 x 64
        out_64 = self.tensor_to_img_64(x_64) # batch_size x 3 x 64 x 64

        x_128 = self.scale_128(x_64) # batch_size x hid_dim x 128 x 128
        out_128 = self.tensor_to_img_128(x_128) # batch_size x 3 x 128 x 128

        out_256 = self.scale_256(x_128) # batch_size x (hid_dim / 2) x 256 x 256
        self.keep_out_256 = out_256 
        out_256 = self.tensor_to_img_256(out_256) # batch_size x 3 x 256 x 256

        if self.deploy:
            return out_256
        else:
            return out_64, out_128, out_256, mean, logsigma


if __name__ == '__main__':
    
    txt = torch.rand((4, 1024))
    noise = torch.rand((4, 100,))
    model_G = Generator(sent_dim=1024, noise_dim=100, emb_dim=128, hid_dim=128)
    o_64, o_128, o_256, mean, logsigma = model_G(txt, noise)
    print(o_64.shape) # 4 x 3 x 64 x 64
    print(o_128.shape) # 4 x 3 x 128 x 128
    print(o_256.shape) # 4 x 3 x 256 x 256