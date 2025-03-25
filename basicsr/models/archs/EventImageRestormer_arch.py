## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # print("x:{}".format(x.shape))
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(in_dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(in_dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) 
        x = x * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'
        b,c,h,w = x.shape
        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(in_dim, LayerNorm_type)
        self.attn = Attention(in_dim,num_heads, bias)
        self.norm2 = LayerNorm(in_dim, LayerNorm_type)
        self.ffn = FeedForward(in_dim, out_dim, ffn_expansion_factor, bias)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # print("x:{}".format(x.shape))
        att_x = self.attn(self.norm1(x))
        x = x + att_x  
        ffn_x = self.ffn(self.norm2(x))
        x = self.conv(x)
        x = x + ffn_x
        return x

##########################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cross_attention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Cross_attention, self).__init__()
        self.norm1= LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x, y):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert x.shape == y.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = x.shape
        fused = x + self.attn(self.norm1(x), self.norm1(y)) # b, c, h, w
        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)
        return fused

class Transformer_EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Transformer_EncoderBlock, self).__init__()
        self.attn = Cross_attention(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.fusion = TransformerBlock(dim*2, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)

    def forward(self, image, event):
        img_feat = self.attn(image, event)
        event_feat = self.attn(event, image)
        ei_feat = torch.cat([img_feat, event_feat], dim=1)
        fusion_feat = self.fusion(ei_feat)
        return img_feat, event_feat, fusion_feat


class Transformer_DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Transformer_DecoderBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))  
        x = x + self.ffn(self.norm2(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class EventImageRestormer(nn.Module):
    def __init__(self, 
        # inp_channels=3, 
        image_in_channels=3,
        event_in_channels=6,
        out_channels=3, 
        dim = 48,
        # num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(EventImageRestormer, self).__init__()

        self.image_patch_embed = OverlapPatchEmbed(image_in_channels, dim)
        self.event_patch_embed = OverlapPatchEmbed(event_in_channels, dim)

        # self.encoder_level1 = nn.Sequential(*[Transformer_EncoderBlock(dim=2*dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1 = Transformer_EncoderBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2, dim---->dim*2
        # self.encoder_level2 = nn.Sequential(*[Transformer_EncoderBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2 = Transformer_EncoderBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3, , 2*dim---->dim*4
        # self.encoder_level3 = nn.Sequential(*[Transformer_EncoderBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3 = Transformer_EncoderBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.latent = nn.Sequential(*[TransformerBlock(in_dim=int(dim*2**3)*2, out_dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent = TransformerBlock(in_dim=int(dim*2**3)*2, out_dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[Transformer_DecoderBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3 = Transformer_DecoderBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[Transformer_DecoderBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2 = Transformer_DecoderBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.decoder_level1 = nn.Sequential(*[Transformer_DecoderBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = Transformer_DecoderBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        self.refinement = nn.Sequential(*[Transformer_DecoderBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, event):
        # 由输入获取embeding
        in_enc_level1_image = self.image_patch_embed(x)
        in_enc_level1_event = self.event_patch_embed(event)
        # 多模态输入组合
        out_image1, out_event1, out_enc_level1 = self.encoder_level1(in_enc_level1_image, in_enc_level1_event)
        # 下采样
        in_enc_level2_image = self.down1_2(out_image1)  # H,W,dim ----> H/2,W/2,dim*2
        in_enc_level2_event = self.down1_2(out_event1)

        out_image2, out_event2, out_enc_level2 = self.encoder_level2(in_enc_level2_image, in_enc_level2_event)

        # 下采样
        in_enc_level3_image = self.down2_3(out_image2)  # H/2,W/2,dim*2 ----> H/4,W/4,dim*4
        in_enc_level3_event = self.down2_3(out_event2)

        out_image3, out_event3, out_enc_level3 = self.encoder_level3(in_enc_level3_image, in_enc_level3_event) 

        # 下采样
        in_enc_level4_image = self.down3_4(out_image3)  # H/4,W/4,dim*4 ----> H/8,W/8,dim*8
        in_enc_level4_event = self.down3_4(out_event3)
  
        in_laent = torch.cat([in_enc_level4_image, in_enc_level4_event], dim=1)
        # print("in_laent:{}".format(in_laent.shape))
        latent = self.latent(in_laent) 
        
        # 上采样
        in_dec_level3 = self.up4_3(latent)  # H/8,W/8,dim*8 ----> H/4,W/4,dim*4
        in_dec_level3 = torch.cat([in_dec_level3, out_enc_level3], 1)  # H/4,W/4,dim*4 ----> H/4,W/4,dim*8
        in_dec_level3 = self.reduce_chan_level3(in_dec_level3)   # # H/4,W/4,dim*8 ----> H/4,W/4,dim*4
        out_dec_level3 = self.decoder_level3(in_dec_level3) 
        # 上采样
        in_dec_level2 = self.up3_2(out_dec_level3)  # H/4,W/4,dim*4 ----> H/2,W/2,dim*2
        in_dec_level2 = torch.cat([in_dec_level2, out_enc_level2], 1)
        in_dec_level2 = self.reduce_chan_level2(in_dec_level2)
        out_dec_level2 = self.decoder_level2(in_dec_level2) 
        # 上采样
        in_dec_level1 = self.up2_1(out_dec_level2)  # H/2,W/2,dim*2 ----> H,W,dim
        in_dec_level1 = torch.cat([in_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(in_dec_level1)
        
        out_refine = self.refinement(out_dec_level1)

        out = self.output(out_refine) + x

        return out

