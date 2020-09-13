import torch
import torch.nn as nn


__all__ = ['HybridCNN']


class HybridCNN(nn.Module):

    def __init__(self, vocab_dim: int, conv_channels, conv_kernels, conv_strides,
                 rnn_hidden_size: int, rnn_num_layers: int, emb_dim=1024, conv_dropout=0.0,
                 rnn_dropout=0.0, lin_dropout=0.0, rnn_bidir=False, lstm=False, map_to_emb=True):

        super().__init__()

        assert hasattr(conv_channels, '__len__')
        assert hasattr(conv_kernels, '__len__')
        assert hasattr(conv_strides, '__len__')
        assert len(conv_channels) == len(conv_kernels)
        assert len(conv_channels) == len(conv_strides)

        assert map_to_emb or (rnn_hidden_size * (1 + int(rnn_bidir)) == emb_dim)

        self.map_to_emb = map_to_emb

        conv_channels = [vocab_dim] + conv_channels

        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s in zip(conv_channels[:-1], conv_channels[1:],
        	                           conv_kernels, conv_strides):
            self.conv_layers.append(
            	nn.Sequential(
            		nn.Conv1d(in_ch, out_ch, k),
            		nn.MaxPool1d(s),
            		nn.ReLU(),
            		nn.Dropout(conv_dropout)))
        if lstm:
            self.rnn = nn.LSTM(conv_channels[-1], hidden_size=rnn_hidden_size,
            	               num_layers=rnn_num_layers, batch_first=True,
            	               dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            	               bidirectional=rnn_bidir)
        else:
            self.rnn = nn.RNN(conv_channels[-1], hidden_size=rnn_hidden_size,
            	              num_layers=rnn_num_layers, batch_first=True,
            	              dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            	              bidirectional=rnn_bidir, nonlinearity='relu')
        if map_to_emb:
        	self.emb_mapper = nn.Sequential(
        		nn.Dropout(lin_dropout),
        		nn.Linear(rnn_hidden_size * (1 + int(rnn_bidir)), emb_dim))


    def _forward(self, x):
    	for conv1 in self.conv_layers:
    	    x = conv1(x)
    	x = self.rnn(x.transpose(1, 2))
    	return x


    def compute_mean_hidden(self, x):
        if self.rnn.bidirectional:
            direction_size = x.size(-1) // 2
            x_front = x[..., :direction_size]
            x_back = x[..., torch.arrange(direction_size * 2 - 1, direction_size - 1, -1)]
            x_ = torch.cat(x_front, x_back, dim=2)
            return x_.mean(dim=1)
        return x.mean(dim=1)


    def forward(self, x):
        assert torch.is_tensor(x)
        assert x.size(1) == next(self.conv_layers[0].children()).in_channels

        x = self._forward(x)
        x = self.compute_mean_hidden(x[0])
        if self.map_to_emb:
            x = self.emb_mapper(x)
        return x


    def name(self):
        return "CRNN"

class TextCNN(nn.Module):
    
    def __init__(self, vocab_dim, text_width, conv_channels, conv_kernels,
    	         conv_strides, emb_dim=1024):
        super().__init__()
        conv_channels = [vocab_dim] + conv_channels

        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s in zip(conv_channels[:-1], conv_channels[1:],
        	                           conv_kernels, conv_strides):
            self.conv_layers.append(
            	nn.Sequential(
            		nn.Conv1d(in_ch, out_ch, k),
            		nn.MaxPool1d(s),
            		nn.ReLU()))

            text_width = (text_width - k + 1) // s

        self.emb_mapper = nn.Linear(conv_channels[-1] * text_width, emb_dim)


    def forward(self, x):
        for conv1 in self.conv_layers:
            x = conv1(x)
        x = x.view(x.size(0), -1)
        x = self.emb_mapper(x)
        return x


if __name__ == '__main__':
    
    text_vector = torch.randn((4, 345, 16))
    # encoder = TextCNN(vocab_dim=345, text_width=16, conv_channels=[128, 256], conv_kernels=[3, 3],
    # 	              conv_strides=[1, 1])
    encoder = HybridCNN(vocab_dim=345, conv_channels=[128, 256], conv_kernels=[3, 3], conv_strides=[1, 1],
                        rnn_hidden_size=256, rnn_num_layers=1,)
    embedding = encoder(text_vector)
    print(embedding.shape)