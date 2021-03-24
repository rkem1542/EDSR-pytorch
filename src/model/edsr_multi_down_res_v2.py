from model import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head1 = [
                nn.MaxPool2d(kernel_size=2, stride=2)
                ]
        m_head2 = [
                conv(args.n_colors, n_feats, kernel_size)
                ] 
        m_head3 = [
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv(n_feats, n_feats, kernel_size),
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv(n_feats, n_feats, kernel_size)
                ]
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, args.n_colors, kernel_size))

        # define tail module
        m_tail_1 = [
            common.Upsampler(conv, 2, args.n_colors, act=False),
            conv(args.n_colors, args.n_colors, kernel_size)
        ]
        m_tail_2 = [
            common.Upsampler(conv, 2, args.n_colors, act=False),
            conv(args.n_colors, n_feats, kernel_size)
        ]
        m_tail_3 = [
            common.Upsampler(conv, 2, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        
       # define another flow of pooling module, pool_depth = 3 
        m_pool_1 = [
                nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        m_pool_2 = [
                nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        self.head1 = nn.Sequential(*m_head1)
        self.head2 = nn.Sequential(*m_head2)
        self.head3 = nn.Sequential(*m_head3)
        self.body = nn.Sequential(*m_body)
        self.tail_1 = nn.Sequential(*m_tail_1)
        self.tail_2 = nn.Sequential(*m_tail_2)
        self.tail_3 = nn.Sequential(*m_tail_3)
        
        self.pool_1 = nn.Sequential(*m_pool_1)
        self.pool_2 = nn.Sequential(*m_pool_2)
    
    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.head1(x)
        x2 = self.head2(x1)
        y = self.head3(x2)
        x3 = self.pool_1(x1)
        x4 = self.pool_2(x3)

        res = self.body(y)
        res += x4

        z1 = self.tail_1(res)
        z1 += x3

        z2 = self.tail_2(z1)
        z2 += x2

        z3 = self.tail_3(z2)
        z3 += x
       
        x = self.add_mean(z3)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

