import sys
sys.path.append('../..')
import torch
import torch.nn as nn

from utils.config import cfg


def conv1x1(in_planes, out_planes):
    return nn.COnv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


def func_attention(query, context, gamma1):
    # query: batch_size x ndf x queryL
    # context: batch_size x ndf x ih x iw
    # mask: batch_size x sourceL
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous() # batch_size x (ih * iw) x ndf

    attn = torch.bmm(contextT, query) # batch_size x (ih * iw) x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    atten = nn.Softmax()(attn)

    attn = attn.view(batch_size, sourceL, queryL)
    attn = attn.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)

    attnT = torch.transpose(attn, 1, 2).contiguous() # batch_size x sourceL x queryL

    weightedContext = torch.bmm(context, attnT) # batch_size ndf x queryL

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GLAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GLAttentionGeneral, self).__init__
        self.conv_context = conv1x1(cdf, idf)
        self.conv_sentence_vis = conv1x1(idf, idf)
        self.linear = nn.Linear(100, idf)
        self.sm = nn.Softmax()
        self.mask = None


    def applyMask(self, mask):
        self.mask = mask # batch_size x sourceL


    def forward(self, input, sentence, context):
        # input: batch_size x idf x ih x iw
        # sentence: batch_size x 100
        # context: batch_size x cdf x sourceL

        idf, ih, iw = input.size(1), input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        target = input.view(batch_size, -1, queryL) # batch_size x idf x (ih * iw)
        targetT = torch.transpose(target, 1, 2).contiguous()

        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3) # batch_size x idf x sourceL

        attn = torch.bmm(targetT, sourceT) # batch_size x (ih * iw) x sourceL
        attn = attn.view(batch_size * queryL, sourceL)

        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1) # (batch_size * queryL) x sourceL
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous() # batch_size x sourceL x queryL

        weightedContext = torch.bmm(sourceT, attn) # batch_size x idf x queryL
        weightedContext = weightedContext.view(batch_size, -1, ih, iw) # batch_size x idf x ih x iw
        word_attn = attn.view(batch_size, -1, ih, iw) # batch_size x sourceL x ih x iw

        sentence = self.linear(sentence) # batch_size x idf
        sentence = sentence.view(batch_size, idf, 1, 1)
        sentence = sentence.repeat(1, 1, ih, iw) # batch_size x idf x ih x iw
        sentence_vs = torch.mul(input, sentence) # batch_size x idf x ih x iw
        sentence_vs = self.conv_sentence_vis(sentence_vs) # batch_size x idf x ih x iw

        sent_att = sentence_vs.view(batch_size * ih * iw, idf)
        sent_att = nn.Softmax()(sent_att)
        sent_att = sent_att.view(batch_size, idf, ih, iw)

        if cfg.MODEL.MIRRORGAN.GLOBAL_ATTEN_ON:
            weightedSentence = torch.mul(sentence, sent_att)
        else:
            weightedSentence = sentence

        return weightedContext, weightedSentence, word_attn, sent_att
    