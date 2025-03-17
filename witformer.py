'''
WIT-Former
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import ipdb
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange, repeat
from torch.nn import init
import matplotlib.pyplot as plt

####### Classes and functiions for adapting windows attention START
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        if num_heads == 0:
            raise ValueError("num_heads must be greater than zero")
        head_dim = max(dim // num_heads, 1)
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1) 
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
      
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B,C, H, W= x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
    return x
####### Classes and functions for adapting windows attention END #####

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        

class eMSM_T(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_T, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.position_embedding=PositionalEncoding(d_model=dim)
        
        self.project_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.)
        )

    def forward(self, x):
        b,c,t,h,w=x.shape

        x=F.adaptive_avg_pool3d(x,(t,1,1))

        x=x.squeeze(-1).squeeze(-1).permute(2,0,1) #t,b,c

        x= self.position_embedding(x).permute(1,0,2) #b,t,c

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        #ipdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))

        scale = (c//self.num_heads) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        out = self.project_out(out).permute(0,2,1)

        return out


class WITFormerBlock(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads_s=8,num_heads_t=2,kernel_size=1,stride=1,padding=0, groups=1,
                win_size = 8, shift_size = 0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., token_projection='linear', shift_flag = False, modulator = False,  #win params
                bias=False,res=True,attention_s=False,attention_t=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res=res
        self.attn_s=attention_s
        self.attn_t=attention_t
        self.num_heads_s=num_heads_s
        self.num_heads_t=num_heads_t
        self.activation=nn.LeakyReLU(inplace=True)
        
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(input_channel)
        
        if shift_flag:
            self.shift_size = self.win_size // 2
        else:
            self.shift_size = shift_size
        
        if modulator:
            self.modulator = nn.Embedding(win_size*win_size, input_channel) # modulator
        else:
            self.modulator = None

        if attention_s==True:
            #self.attention_s=eMSM_I(dim=input_channel, num_heads=num_heads_s, bias=False)
            self.win_attn = WindowAttention(
            dim = input_channel, win_size=to_2tuple(self.win_size), num_heads=num_heads_t,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection) 
            
        self.conv_1x3x3=nn.Conv3d(input_channel,output_channel,kernel_size=(1, kernel_size[1], kernel_size[2]),
                            stride=(1, stride[1], stride[2]),padding=(0, padding[1], padding[2]),groups=groups)
        if attention_t==True:
            self.attention_t=eMSM_T(dim=input_channel, num_heads=num_heads_t, bias=False)
                       
            

        self.conv_3x1x1=nn.Conv3d(input_channel,output_channel,kernel_size=(kernel_size[0], 1, 1),
                            stride=(stride[0], 1, 1),padding=(padding[0], 0, 0),groups=groups)
        
        

        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        

    def forward(self, inputs, mask =None):
        
        ####### win att start
        B, C, D, H, W = inputs.shape

        if self.attn_s==True or self.attn_t==True:
             ## input mask
            if mask != None:
                input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)#!!! 
                input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
                attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
                attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
                attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            input_x = F.adaptive_avg_pool3d(inputs, (1, H, W))  # x: (b, c, 1, h, w)
            input_x = input_x.permute(0, 1, 3, 4, 2).squeeze(-1)  # x: (b, c, h, w)
            ## shift mask
            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                shift_mask = torch.zeros((1, H, W, 1)).type_as(input_x)
                h_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1
                shift_mask=shift_mask.permute(0,3,1,2)
                shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
                shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
                shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
                shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
                attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
            
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(input_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = input_x

            # partition windows
            x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
            x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
            
             # with_modulator
            if self.modulator is not None:
                wmsa_in = self.with_pos_embed(x_windows,self.modulator.weight)
            else:
                wmsa_in = x_windows

            # W-MSA/SW-MSA
            attn_windows = self.win_attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B C H' W' 

            # reverse cyclic shift
            if self.shift_size > 0:
                attn_w = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                attn_w = shifted_x
            #x = x.view(B, H * W, C)

            ####### win att end

            #attn_s=self.attention_s(inputs).unsqueeze(2)  if self.attn_s==True else 0 
            
            #layer normalization
            #attn_w=shortcut+attn_w
            
            attn_w=attn_w.unsqueeze(2)  if self.attn_s==True else 0 
            attn_t=self.attention_t(inputs).unsqueeze(-1).unsqueeze(-1) if self.attn_t==True else 0

            #inputs_attn=inputs+attn_t+attn_s
            inputs_attn=inputs+attn_t+attn_w

            conv_S=self.conv_1x3x3(inputs_attn)
            conv_T=self.conv_3x1x1(inputs_attn)

            if self.input_channel == self.output_channel: 
                identity_out=inputs_attn 
            else: 
                identity_out=self.shortcut(inputs_attn)

        else:
            if self.input_channel == self.output_channel: 
                identity_out=inputs 
            else: 
                identity_out=self.shortcut(inputs)
                
            conv_S=self.conv_1x3x3(inputs)
            conv_T=self.conv_3x1x1(inputs)

        if self.res:
            output=conv_S+conv_T+identity_out
        elif not self.res:
            output=conv_S+conv_T  

        return output


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False, shift_flag = False):
        super(DoubleConv,self).__init__()
        self.double_conv=nn.Sequential(
            WITFormerBlock(in_channels,in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                            attention_s=attention_s,attention_t=attention_t, shift_flag = True),
            nn.LeakyReLU(inplace=True),
            WITFormerBlock(in_channels,out_channels,res=res),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                 res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                       res=res,attention_s=attention_s,attention_t=attention_t)
        )
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            WITFormerBlock(in_channels,2*in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                      attention_s=attention_s,attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            WITFormerBlock(2*in_channels,out_channels),
            nn.LeakyReLU(inplace=True),
            )
    def forward(self, x):
        return self.encoder(x)


    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.res_unet=res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                               res=res,attention_s=attention_s,attention_t=attention_t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,res=True,activation=False):
        super().__init__()
        self.act=activation
        self.conv =WITFormerBlock(in_channels, out_channels,res=res)
        self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        
        
class WITFormer(nn.Module):
    def __init__(self, in_channels,out_channels,n_channels,num_heads_s=[1,2,4,8],num_heads_t=[1,2,4,8],
                win_size = 8, shift_size = 0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., token_projection='linear', shift_flag=False, 
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation=True)
        self.enc1 = DoubleConv(n_channels//2, n_channels,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t, shift_flag=shift_flag) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[3],num_heads_t=num_heads_t[3],
                             res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=True)
        self.depth_up = nn.Upsample(scale_factor=tuple([2.5,1,1]),mode='trilinear')
        #self.depth_up = nn.Upsample(scale_factor=tuple([2,1,1]),mode='trilinear')
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation=False)

    def forward(self, x):
        b,c,d,h,w=x.shape
        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)+x
        output = self.depth_up(output)
        output = self.out2(output)
        return output

