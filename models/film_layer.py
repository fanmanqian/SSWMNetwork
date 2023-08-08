# @Author : cheertt
# @Time   : 19-11-5 下午2:40
# @Remark : film_layer
import torch
import torch.nn as nn
from torch.autograd import Variable


class FilmLayer(nn.Module):
    """
        A very basic FiLM layer with a linear transformation from context to FiLM parameters
    """
    def __init__(self):
        super(FilmLayer, self).__init__()
        # (batch_size, channels, height, width)
        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.feature_size = None

        self.fc = DynamicFC().cuda()

    def forward(self, feature_maps, context):
        """
            Arguments:
                feature_maps : input feature maps (N,C,H,W)
                context : context embedding (N,L)
            Return:
                output : feature maps modulated with betas and gammas (FiLM parameters)
        """
        self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
        # FiLM parameters needed for each channel in the feature map
        # hence, feature_size defined to be same as no. of channels
        self.feature_size = feature_maps.data.shape[1]

        # linear transformation of context to FiLM parameters
        film_params = self.fc(context, out_planes=2 * self.feature_size, activation=None)

        # stack the FiLM parameters across the spatial dimension
        film_params = torch.stack([film_params] * self.height, dim=2)
        film_params = torch.stack([film_params] * self.width, dim=3)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, :self.feature_size, :, :]
        betas = film_params[:, self.feature_size:, :, :]

        # modulate the feature map with FiLM parameters
        output = (1 + gammas) * feature_maps + betas

        return output


class DynamicFC(nn.Module):
    """
        dynamic FC - for varying dimensions on the go
        Input - embedding in (batch_size(N), * , channels(C)) [* represent extra dimensions]
    """
    def __init__(self):
        super(DynamicFC, self).__init__()

        self.in_planes = None
        self.out_planes = None
        self.activation = None
        self.use_bias = None

        self.activation = None
        self.linear = None

    def forward(self, embedding, out_planes=1, activation=None, use_bias=True):
        """
        :param embedding: input to the MLP (N,*,C)
        :param out_planes: total channels in the output
        :param activation: 'relu' or 'tanh'
        :param use_bias: True / False
        :return:
                out : output of the MLP (N,*,out_planes)
        """
        self.in_planes = embedding.data.shape[-1]
        self.out_planes = out_planes
        self.use_bias = use_bias

        self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias).cuda()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True).cuda()
        elif activation == 'tanh':
            self.activation = nn.Tanh().cuda()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if self.use_bias:
                    nn.init.constant_(m.bias, 0.1)

        out = self.linear(embedding)
        if self.activation is not None:
            out = self.activation(out)

        return out