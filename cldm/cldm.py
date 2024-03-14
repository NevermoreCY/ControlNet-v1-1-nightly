import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer, SpatialTransformer3D
from ldm.modules.diffusionmodules.openaimodel import UNetModel,  TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock , MultiViewUNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

DEBUG =False
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class MultiViewControlledUnetModel(MultiViewUNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, global_emb=None, **kwargs):
        hs = []


        # with torch.no_grad():
            # t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            # emb = self.time_embed(t_emb)


        # print("global embd is ", global_emb)
        emb = global_emb
        h = x.type(self.dtype)

        # print('\n\n\n h shape', h.shape, 'emb : ', emb, 'context : ', context.shape)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)


        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

    # def forward_ex(self, x, timesteps=None, context=None, y=None, camera=None, num_frames=1, **kwargs):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param context: conditioning plugged in via crossattn
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :param num_frames: a integer indicating number of frames for tensor reshaping.
    #     :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
    #     """
    #
    #     assert x.shape[0] % num_frames == 0, "[UNet] input batch size must be dividable by num_frames!"
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"
    #     hs = []
    #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #     emb = self.time_embed(t_emb)
    #
    #     if self.num_classes is not None:  # Is None
    #         assert y.shape[0] == x.shape[0]
    #         emb = emb + self.label_emb(y)
    #
    #     # Add camera embeddings
    #     if camera is not None:
    #         assert camera.shape[0] == emb.shape[0]
    #         emb = emb + self.camera_embed(camera)
    #
    #     DEBUG =False
    #     if DEBUG:
    #         print("\n\n\n\n\n forward of multiview unet, x shape", x.shape)  # [8,4,32,32]
    #         print("\n  camera shape", camera.shape)  # [8,16]
    #         print("\n  timesteps shape", timesteps.shape)  # [8]
    #         print("\n  t_emb shape", t_emb.shape)  # [8, 320]
    #         print("\n  y ", y )  # None
    #         print("\n  emb shape", emb.shape)  #  [8,1280]
    #         print("\n  context shape", context.shape)  # [8,77,1024]
    #
    #     h = x.type(self.dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb, context, num_frames=num_frames)
    #         hs.append(h)
    #     h = self.middle_block(h, emb, context, num_frames=num_frames)
    #     for module in self.output_blocks:
    #         h = th.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb, context, num_frames=num_frames)
    #     h = h.type(x.dtype)
    #     if self.predict_codebook_ids:
    #         return self.id_predictor(h)
    #     else:
    #         return self.out(h)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    print("num_attention_blocks should be None, which is : " , num_attention_blocks)
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class MultiViewControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            camera_dim = None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # v1
        # control_dim = 256 * 32 * 32
        # print("mlp1 size: ", control_dim)
        # self.zero_mlp1 = nn.Sequential(
        #     linear(time_embed_dim, control_dim),
        #     nn.SiLU(),
        # )

        # v3
        # control_dim = image_size * image_size
        # self.zero_mlp1 = nn.Sequential(
        #     zero_module(linear(time_embed_dim, control_dim)),
        #     nn.SiLU(),
        #     zero_module(linear(control_dim, control_dim)),
        # )

        # V3
        # control_dim_v3 = image_size * image_size * 256
        # self.zero_mlp2 =  nn.Sequential(
        #     zero_module(linear(control_dim_v3, time_embed_dim)),
        #     nn.SiLU(),
        #     zero_module(linear(time_embed_dim, time_embed_dim)),
        # )

        # v4:
        # self.global_emb_conv = nn.Sequential(
        #     conv_nd(dims, 320, 160, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 160, 160, 3, padding=1, stride=4),
        #     nn.SiLU(),
        #     conv_nd(dims, 160, 80, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 80, 80, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, 80, 80, 3, padding=1)),
        #     nn.SiLU(),
        # )

        # version proposed by author
        #  mlp1
        self.zero_mlp1 = zero_module(linear(time_embed_dim, model_channels))

        #  mlp2

        conv_1x1 = nn.Sequential(
            nn.Conv2d(model_channels, time_embed_dim , kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(time_embed_dim, time_embed_dim, kernel_size=1),
        )
        self.zero_mlp2 = zero_module(conv_1x1)




        # add camera embd
        if camera_dim is not None:
            # self.camera_embed_pre = nn.Sequential(
            #     linear(12, camera_dim),
            # )

            self.camera_embed = nn.Sequential(
                linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )


        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # B , 320 , 32 , 32

        self.hint_mixed_conv_out = TimestepEmbedSequential(
            zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
        )

        # self.channel_compress = TimestepEmbedSequential(
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )



        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    print("num_attention_blocks should be None, which is : " , num_attention_blocks)
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer3D(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch)) # extra zero conv
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer3D(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context,camera=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        # DEBUG = True

        if DEBUG:
            print("\n\n\n hint original shape: " , hint.shape)

        guided_hint = self.input_hint_block(hint, emb, context)

        if DEBUG:
            print("\n Guided_hint shape: " , guided_hint.shape)

        # hint original shape:  torch.Size([160, 3, 256, 256])
        #  Guided_hint shape:  torch.Size([160, 320, 32, 32])

        if camera is not None:
            # check batch size
            assert camera.shape[0] == emb.shape[0]
            # camera = self.camera_embed_pre(camera)
            emb_res = emb + self.camera_embed(camera)

        # print("\n camera + t  embedding shape is : ", emb.shape)
        # camera + t  embedding shape is :  torch.Size([120, 1280])

        emb = self.zero_mlp1(emb_res)

        # print("\n zero mlp1 emb : ", emb.shape)

        #--------------v2------------------
        # gh0,gh1,gh2,gh3 = guided_hint.shape
        # emb = rearrange(emb, "b (c h w) -> b c h w", c=256,h=self.image_size,w=self.image_size).contiguous()
        # print("\n zero mlp1 emb after rearrange : ", emb.shape)
        #-------------v2---------------


        # V3 repeat emb for num_channel times



        emb = emb[:,:,None,None]
        # print("\n emb add 2 dim : ", emb.shape)
        emb = emb.repeat(1,1,self.image_size,self.image_size)
        # print("\n emb repeat model channels times : ", emb.shape)
        # emb = rearrange(emb, "b c (h w) -> b c h w", h=self.image_size,w=self.image_size).contiguous()
        # print("\n emb rearrange to match image size  : ", emb.shape)

        cond_with_camera_t = guided_hint + emb
        # print("\n cond_with_camera_t rearrange : ", cond_with_camera_t.shape)

        cond_with_camera_t = self.hint_mixed_conv_out(cond_with_camera_t,emb,context)
        # print("\n ~~cond_with_camera_t conv out : ", cond_with_camera_t.shape)
        #  ~~cond_with_camera_t conv out :  torch.Size([120, 320, 32, 32])

        local_emb = cond_with_camera_t

        # global_emb = self.global_emb_conv(cond_with_camera_t)

        global_emb = self.zero_mlp2(cond_with_camera_t)

        # global_emb = rearrange(emb, "b c h w -> b (c h w)").contiguous()
        # print("\n zero mlp2 emb after global emb conv: ", global_emb.shape)
        # global_emb = self.zero_mlp2(global_emb)

        # v4
        global_emb = rearrange(global_emb, "b c h w -> b c (h w)").contiguous()
        # print("\n zero mlp2 emb after rearrange : ", global_emb.shape)
        global_emb = torch.sum(global_emb, dim=2)
        # print("\n glob  emb after sum up : ", global_emb.shape)

        # residule for emb
        # global_emb = global_emb + emb_res



        outs = []

        h = x.type(self.dtype)

        # print("\n h shape is : ", h.shape)
        # print("\n Context shape is : ", context.shape)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if local_emb is not None:
                h = module(h, global_emb, context)

                # print('\n after first module, h shape is : ', h.shape)
                h += local_emb
                local_emb = None
            else:
                h = module(h, global_emb, context)
            outs.append(zero_conv(h, global_emb, context))

        h = self.middle_block(h, global_emb, context)
        outs.append(self.middle_block_out(h, global_emb, context))

        return outs , global_emb


class MultiViewControlNet_v0(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            camera_dim = None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # add camera embd
        if camera_dim is not None:
            # self.camera_embed_pre = nn.Sequential(
            #     linear(12, camera_dim),
            # )

            self.camera_embed = nn.Sequential(
                linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )


        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # B , 320 , 32 , 32

        # self.hint_mixed_conv_out = TimestepEmbedSequential(
        #     zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
        # )

        # self.channel_compress = TimestepEmbedSequential(
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )



        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    print("num_attention_blocks should be None, which is : " , num_attention_blocks)
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer3D(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch)) # extra zero conv
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer3D(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context,camera=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        # DEBUG = True

        if DEBUG:
            print("\n\n\n hint original shape: " , hint.shape)

        guided_hint = self.input_hint_block(hint, emb, context)

        if DEBUG:
            print("\n Guided_hint shape: " , guided_hint.shape)

        # hint original shape:  torch.Size([160, 3, 256, 256])
        #  Guided_hint shape:  torch.Size([160, 320, 32, 32])

        if camera is not None:
            # check batch size
            assert camera.shape[0] == emb.shape[0]
            # camera = self.camera_embed_pre(camera)
            emb = emb + self.camera_embed(camera)

        global_emb = emb




        outs = []

        h = x.type(self.dtype)

        # print("\n h shape is : ", h.shape)
        # print("\n Context shape is : ", context.shape)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, global_emb, context)

                # print('\n after first module, h shape is : ', h.shape)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, global_emb, context)
            outs.append(zero_conv(h, global_emb, context))

        h = self.middle_block(h, global_emb, context)
        outs.append(self.middle_block_out(h, global_emb, context))

        return outs , global_emb


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, global_average_pooling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.global_average_pooling = global_average_pooling

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):

        # print("* args and **kwargs should be None!" , "*args are" , *args, "**kwargs are", **kwargs)
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # if DEBUG:
        #     print("before rearrange, x shape is ", x.shape , ' c shape is ', c.shape )
            #before rearrange, x shape is  torch.Size([160, 4, 32, 32])  c shape is  torch.Size([160, 77, 768])


        # get control image
        control = batch[self.control_key]
        # get camera info
        T = batch['camera_pose'].to(memory_format=torch.contiguous_format).float()

        # DEBUG=True

        if DEBUG:
            print("\n\n\n Before rearrange: control shape is ", control.shape )  # torch.Size([160, 3, 256, 256])
            print("\n shape of T is ", T.shape)     # torch.Size([40,4, 3, 4])

        T = rearrange(T, "b f x -> (b f) x").contiguous()
        control = rearrange(control, "b f h w c -> (b f) h w c").contiguous()

        if DEBUG:
            print("\n\n\n after rearrange: control shape is ", control.shape )
            print("\n shape of T is ", T.shape)


        if bs is not None:
            control = control[:bs]
            T = T[:bs]

        T = T.to(self.device)
        control = control.to(self.device)

        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control],camera=[T])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        camera = cond['camera'][0]
        if DEBUG:
            print('\n cond_txt shape : ' ,cond_txt.shape)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # print('\n ~~~~yeah~')
            control , global_emb = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, camera=camera)
            control = [c * scale for c, scale in zip(control, self.control_scales)]

            # print('\n\n ***** control model finished!')
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

            # print("\n diffsuion model start!")
            # print("\n\n Input global_emb shape : " ,global_emb.shape)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control , global_emb=global_emb)
            # print("\n diffusion model ends!")
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # z noised label image , c is control + camera + text


        DEBUG = True

        if DEBUG:
            print('\n\n\n  !!camera: ', c['camera'])

        c_cat, c ,camera = c["c_concat"][0][:N], c["c_crossattn"][0][:N] ,c["camera"][0][:N]

        if DEBUG:
            print("\n\n c_cat : " , c_cat.shape , 'c_cross : ' , c.shape , "camera : " ,  camera.shape)


        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0


        # batch_text = batch[self.cond_stage_key]

        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c] , 'camera':[camera]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            if DEBUG:
                print("\n\n\n unconditional_guidance_scale > 1.0:")
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross] , 'camera':[camera]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c] , 'camera' : [camera]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            # print("\n\n x_samples_cfg shape is : ", x_samples_cfg.shape)
            #  x_samples_cfg shape is :  torch.Size([32, 3, 256, 256])
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        # params1 = list(self.control_model.parameters())
        model_dict = self.control_model.state_dict()
        params = []

        for k in model_dict.keys():
            # print('\n',k)
            if 'camera_embed.' in k:
                # print('\n Found camera model ')
                model_dict[k].requires_grad = False
                # print(model_dict[k])
                params.append(model_dict[k])
            else:
                DEBUG=True

                # if DEBUG:
                #     print("\n\n Before set ", k , model_dict[k].requires_grad)

                model_dict[k].requires_grad = True

                # if DEBUG:
                #     print("\n\n After set ", k , model_dict[k].requires_grad)

                params.append(model_dict[k])

        # print(len(params) , len(params1))

        # DEBUG = True
        # if DEBUG:
        #     print("\n\n\n check optimizing parameters")
        #     for item in params:
        #         print("\n\n\n" , item)
        # print("\n\n\n\n")
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())


        # for item in params:
        #     print("\n",item.requires_grad)

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    # need lock camera embed in control model
    # control_model.camera_embed_pre.0.weight
    # control_model.camera_embed_pre.0.bias
    # control_model.camera_embed.0.weight
    # control_model.camera_embed.0.bias
    # control_model.camera_embed.2.weight
    # control_model.camera_embed.2.bias

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
