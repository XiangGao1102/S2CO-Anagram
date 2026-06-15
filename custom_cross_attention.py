import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import random
import torch.nn.functional as F
from typing import Optional, Callable
import math


class CustomCrossAttention(torch.nn.Module):
    """自定义交叉注意力层 - 包装器模式"""

    def __init__(self, original_attention_module, layer_name):
        super().__init__()
        self.original_module = original_attention_module

        # 用于自定义处理的钩子
        self.custom_hooks = {
            'pre_attention': None,  # 在注意力计算前的处理
            'attention_map': None,  # 注意力图的自定义处理
            'post_attention': None  # 在注意力计算后的处理
        }

    def set_custom_hook(self, hook_name: str, hook_func: Optional[Callable]):
        """设置自定义处理钩子"""
        if hook_name in self.custom_hooks:
            self.custom_hooks[hook_name] = hook_func
        else:
            raise ValueError(f"无效的钩子名称: {hook_name}。可选: {list(self.custom_hooks.keys())}")

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        """
        包装器前向传播
        可以自定义注意力机制
        """
        # # 检查是否有自定义钩子
        has_custom_hooks = any(hook is not None for hook in self.custom_hooks.values())

        if not has_custom_hooks:
            return self.original_module(hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)
        
        batch_size, sequence_length, _ = hidden_states.shape

        # 获取query
        query = self.original_module.to_q(hidden_states)
        key = self.original_module.to_k(encoder_hidden_states)
        value = self.original_module.to_v(encoder_hidden_states)

        # 提取注意力参数
        if hasattr(self.original_module, 'heads'):
            heads = self.original_module.heads
        else:
            # 推断头数
            if query.shape[-1] % 64 == 0:
                heads = query.shape[-1] // 64
            else:
                heads = 8
        dim_head = query.shape[-1] // heads

        # 重新塑造为多头
        query = query.view(batch_size, -1, heads, dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, heads, dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, heads, dim_head).transpose(1, 2)

        # 自定义预处理钩子
        if self.custom_hooks['pre_attention'] is not None:
            query, key, value = self.custom_hooks['pre_attention'](query, key, value)

        # 计算注意力分数
        scale = 1.0 / torch.sqrt(torch.tensor(dim_head, dtype=query.dtype, device=query.device))
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

        # 自定义注意力图钩子
        if self.custom_hooks['attention_map'] is not None:
            attention_scores = self.custom_hooks['attention_map'](attention_scores, query, key, value)

        # 应用softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 计算上下文
        context = torch.matmul(attention_probs, value)

        # 自定义后处理钩子
        if self.custom_hooks['post_attention'] is not None:
            context = self.custom_hooks['post_attention'](context, attention_probs)

        # 调整形状
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, heads * dim_head)
        
        # 输出投影 - 处理不同的to_out结构
        if isinstance(self.original_module.to_out, torch.nn.Sequential):
            hidden_states = self.original_module.to_out[0](context)
        elif isinstance(self.original_module.to_out, torch.nn.ModuleList):
            # ModuleList情况，通常只有一个元素
            if len(self.original_module.to_out) > 0:
                hidden_states = self.original_module.to_out[0](context)
            else:
                # 如果没有元素，直接返回上下文
                hidden_states = context
        elif callable(self.original_module.to_out):
            # 可调用对象
            hidden_states = self.original_module.to_out(context)
        else:
            # 其他情况，直接返回上下文
            hidden_states = context

        return hidden_states


def replace_cross_attention_layers(unet, custom_attention_class=CustomCrossAttention):
    """
    更安全地替换UNet中的交叉注意力层
    """
    replaced_layers = {}

    def replace_in_module(module, name_path=""):
        for name, child in module.named_children():
            full_name = f"{name_path}.{name}" if name_path else name
            
            if hasattr(child, 'attn2'):  # 交叉注意力
                replaced_layers[f"{full_name}.attn2"] = child.attn2
                child.attn2 = custom_attention_class(child.attn2, f"{full_name}.attn2")
            else:
                # 递归处理
                replace_in_module(child, full_name)

    replace_in_module(unet)
    return replaced_layers


def restore_cross_attention_layers(unet, original_layers):
    """恢复原始注意力层"""
    for full_name, original_layer in original_layers.items():
        # 解析模块路径
        parts = full_name.split('.')
        module = unet
        for part in parts[:-1]:
            if part:  # 跳过空部分
                module = getattr(module, part)

        # 恢复原始层
        setattr(module, parts[-1], original_layer)


def setup_custom_attention_hooks(unet, hook_config=None):
    """设置自定义钩子"""
    if hook_config is None:
        return []

    def find_custom_layers(module):
        layers = []
        for child in module.children():
            if isinstance(child, CustomCrossAttention):
                layers.append(child)
            else:
                layers.extend(find_custom_layers(child))
        return layers

    custom_layers = find_custom_layers(unet)

    for layer in custom_layers:
        for hook_name, hook_func in hook_config.items():
            if hook_func is not None:
                layer.set_custom_hook(hook_name, hook_func)

    print(f"为 {len(custom_layers)} 个注意力层设置了自定义钩子")
    return custom_layers
