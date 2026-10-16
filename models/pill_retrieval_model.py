import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
import timm
import os, sys

# Make sure the local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pooling.pooling import GeMPooling, MPNCOV
from models.model_category_config import (
    super_large_backbones, large_backbones, medium_backbones,
    small_backbones, vit_backbones, swin_backbones, lightweight_vit_backbones,
    dinov2_backbones
)

class PillRetrievalModel(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50', pooling_type='gem', embedding_size=512, s=64.0, m=0.35):
        super(PillRetrievalModel, self).__init__()
        self.s = s 
        self.m = m 
        self.pooling_type = pooling_type.lower()
        
        # =====================================================
        # 1. CLASSIFY THE ARCHITECTURE WITH AN EXPLICIT MAPPING
        # =====================================================
        clean_name = backbone_type.lower().replace('_tv', '').replace('_timm', '')
        
        # Use 'in' rather than 'any()' so that the name matches the config file exactly
        self.is_vit_backbone = clean_name in vit_backbones
        self.is_swin_backbone = clean_name in swin_backbones
        self.is_lightweight_vit = clean_name in lightweight_vit_backbones
        
        # Pure ViTs (DINOv2, Swin, EVA) return a 2D tensor (Batch, Channels) from timm
        # Hybrid networks and CNNs return a 4D tensor (Batch, Channels, H, W)
        self.is_pure_vit = self.is_vit_backbone or self.is_swin_backbone

        # =====================================================
        # 2. BUILD THE BACKBONE
        # =====================================================
        self.features = self._build_backbone(backbone_type)

        # =====================================================
        # 3. FIND THE NUMBER OF OUTPUT CHANNELS (Dynamic Shape Detection)
        # =====================================================
        self.features.eval() 
        with torch.no_grad():

            # Choose the dummy input size automatically to avoid Patch/Position Embedding errors
            if '_224' in clean_name or self.is_lightweight_vit or clean_name in small_backbones:
                self.safe_size = 224
            elif '_384' in clean_name:
                self.safe_size = 384
            elif 'patch14' in clean_name:
                self.safe_size = 392 # Divisible by 14 (DINOv2 / EVA)
            else:
                self.safe_size = 384
                
            dummy_input = torch.zeros(1, 3, self.safe_size, self.safe_size)
            dummy_out = self.features(dummy_input)
            
            if isinstance(dummy_out, tuple): 
                dummy_out = dummy_out[0]
            
            in_channels = dummy_out.shape[1]
                
        self.features.train() 
        print(f"Model Info | Backbone: {clean_name} | Channels: {in_channels} | "
              f"Output: {dummy_out.dim()}D | Pooling: {'Bypass' if dummy_out.dim() == 2 else self.pooling_type.upper()}")

        # =====================================================
        # 4. PROJECTION HEAD DESIGN (AGAINST FEATURE COLLAPSE)
        # =====================================================
        if self.pooling_type == 'mpncov' and dummy_out.dim() == 4:
            self.mpncov_bottleneck = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
            self.pool = MPNCOV(iterNum=3)
            self.fc_projection = nn.Linear(256 * 256, embedding_size, bias=False)
        else:
            self.pool = GeMPooling(p=3.0)

            # For pure Transformers: an MLP head with LayerNorm and GELU
            if self.is_pure_vit or self.is_lightweight_vit:
                print(f"MLP projection enabled for Transformer/Hybrid: {in_channels} -> {embedding_size}")
                # If the raw feature is too small, widen it right away
                if in_channels < embedding_size:
                    self.fc_projection = nn.Sequential(
                        nn.Linear(in_channels, embedding_size, bias=False),
                        nn.LayerNorm(embedding_size), 
                        nn.GELU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(embedding_size, embedding_size, bias=False)
                    )
                else:
                    self.fc_projection = nn.Sequential(
                        nn.Linear(in_channels, in_channels, bias=False),
                        nn.LayerNorm(in_channels), 
                        nn.GELU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channels, embedding_size, bias=False)
                    )
                    
            # For very large CNNs: a bottleneck that reduces the dimension smoothly
            elif in_channels >= 1024:
                print(f"Bottleneck projection enabled for CNN: {in_channels} -> 1024 -> {embedding_size}")
                self.fc_projection = nn.Sequential(
                    nn.Linear(in_channels, 1024, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.GELU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(1024, embedding_size, bias=False)
                )
            else:
                self.fc_projection = nn.Linear(in_channels, embedding_size, bias=False)

        # =====================================================
        # 5. METRIC LEARNING HEAD (ArcFace / CosFace)
        # =====================================================
        self.bn_head = nn.BatchNorm1d(embedding_size)
        self.fc_ce = nn.Linear(embedding_size, num_classes, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _build_backbone(self, backbone_type):
        raw_name = backbone_type.lower()
        force_tv = raw_name.endswith('_tv')
        clean_name = raw_name.replace('_tv', '').replace('_timm', '')

        if force_tv:
            return self._load_torchvision(clean_name)
        
        return self._load_timm(clean_name)

    def _load_torchvision(self, tv_name):
        # SPECIAL CASE: map the config name onto the matching torchvision function name
        name = tv_name
        if tv_name == 'mobilenetv3_large_100': 
            name = 'mobilenet_v3_large'
        elif tv_name == 'mobilenetv2_100': 
            name = 'mobilenet_v2'

        base_model_func = getattr(models, name)
        base = base_model_func(weights='DEFAULT')
        
        if 'resnet' in name:
            if name in ['resnet18', 'resnet34']: base.layer4[0].conv1.stride = (1, 1)
            else: base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            return nn.Sequential(*list(base.children())[:-2])
            
        if hasattr(base, 'features'):
            return base.features
        return nn.Sequential(*list(base.children())[:-2])
    
    def _load_timm(self, name):
        is_dinov2 = name in dinov2_backbones

        # 1. Set DropPath automatically
        # DINOv2 uses drop_path=0.1 to preserve its pretrained features
        # Super_large/large use 0.3 (down from 0.4): the "small > base > large" pattern
        # in ConvNeXt shows that 0.4 over-corrects for overfitting on the pill dataset
        if is_dinov2:
            drop_path = 0.1
        elif name in super_large_backbones or name in large_backbones:
            drop_path = 0.3
        else:
            drop_path = 0.2

        # 2. Prepare the dictionary of basic arguments
        model_kwargs = {
            'pretrained': True,
            'num_classes': 0,
            'drop_path_rate': drop_path
        }
        # SPECIAL CASE: remove drop_path_rate for older edge CNNs that do not support it
        if any(x in name for x in ['ghostnet', 'mobilenet', 'shufflenet', 'squeezenet', 'repvgg']):
            model_kwargs.pop('drop_path_rate', None)

        # 3. Route the arguments EXPLICITLY
        if self.is_swin_backbone:
            model_kwargs['global_pool'] = 'avg'
            model_kwargs['dynamic_img_size'] = True

        elif 'maxvit' in name:
            # PROTECT MAXVIT: return 4D for GeM pooling, and never pass dynamic_img_size
            model_kwargs['global_pool'] = ''

        elif self.is_vit_backbone:
            # DINOv2 uses 'token' to exploit the [CLS] token; the other ViTs use 'avg'
            if is_dinov2:
                model_kwargs['global_pool'] = 'token'
                print("Keeping the native [CLS] token pooling for DINOv2")
            else:
                model_kwargs['global_pool'] = 'avg'
            model_kwargs['dynamic_img_size'] = True

        elif self.is_lightweight_vit:
            model_kwargs['global_pool'] = 'avg'

        else:
            # Default for every CNN (ResNet, EfficientNet, ConvNeXt, ...)
            model_kwargs['global_pool'] = ''

        # 4. Create the model with the cleaned set of arguments
        model = timm.create_model(name, **model_kwargs)

        # 5. Enable gradient checkpointing to save VRAM on large networks
        # KEEP it for DINOv2: it is needed to fit in 12GB of VRAM; timm's checkpointing
        # saves the RNG state, so the drop_path masks agree between forward and recompute
        if name in super_large_backbones or name in large_backbones:
            try:
                model.set_grad_checkpointing(True)
                print(f"Gradient checkpointing enabled for [{name}]")
            except: pass

        return model

    def forward(self, x, labels=None):
        # Force the safe input size for edge ViTs and small CNNs to avoid tensor shape errors
        if x.shape[-1] != self.safe_size:
             x = F.interpolate(x, size=(self.safe_size, self.safe_size), mode='bilinear', align_corners=False)
             
        feat = self.features(x)
        
        # POOLING BRANCH
        if feat.dim() == 2:
            # Pure ViTs (DINOv2/Swin) already give a 2D tensor, so pooling is skipped
            pooled_feat = feat
        else:
            # CNNs and hybrid ViTs (4D) go through GeM or MPNCOV
            if self.pooling_type == 'mpncov' and not self.is_pure_vit:
                feat = self.mpncov_bottleneck(feat) 
                pooled_feat = self.pool(feat).view(feat.size(0), -1) 
            else:
                pooled_feat = self.pool(feat).view(feat.size(0), -1) 
        
        # Projection & Normalization
        embedding = self.bn_head(self.fc_projection(pooled_feat))
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            # SCE loss branch
            weight_norm = F.normalize(self.fc_ce.weight, p=2, dim=1)
            cosine_sce = F.linear(norm_embedding, weight_norm)
            logits_sce = cosine_sce * self.s
            
            # CSCE loss branch (ArcFace)
            cosine = F.linear(norm_embedding, F.normalize(self.weight))
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7) 
            
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m 
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            
            logits_csce = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            logits_csce *= self.s
            
            return logits_sce, logits_csce, norm_embedding
        
        return norm_embedding