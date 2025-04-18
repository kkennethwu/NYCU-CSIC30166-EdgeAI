from hqq.core.quantize import BaseQuantizeConfig

from matplotlib import pyplot as plt
import numpy as np
def plot_weight_distribution(model, bitwidth):
    
    def get_quantized_range(bitwidth):
        quantized_max = (1 << (bitwidth - 1)) - 1
        quantized_min = -(1 << (bitwidth - 1))
        return quantized_min, quantized_max
    
    # bins = (1 << bitwidth) if bitwidth <= 8 else 256
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = 'left'
    else:
        bins = 256
        align = 'mid'
    fig, axes = plt.subplots(9,17, figsize=(160, 120))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                    align=align, color = 'blue', alpha = 0.5,
                    edgecolor='black' if bitwidth <= 4 else None)
            if bitwidth <= 4:
                quantized_min, quantized_max = get_quantized_range(bitwidth)
                ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle(f'Histogram of Weights (bitwidth={bitwidth} bits)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(f'figs/weight_distribution_{bitwidth}.png')
    plt.close()
            
            


# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model, nbits, group_size):
    # plot_weight_distribution(model, 32)
    
    quant_config = {}
    
    n_blocks = len(model.blocks)
    
    # best: (4, 64, 89.6, 14.196182250976562, 18.80381774902338)
    attn_config = BaseQuantizeConfig(nbits=4, group_size=32)
    proj_config = BaseQuantizeConfig(nbits=4, group_size=64)
    fc1_config = BaseQuantizeConfig(nbits=4, group_size=64)
    fc2_config = BaseQuantizeConfig(nbits=4, group_size=64)
    
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = attn_config
        quant_config[f'blocks.{i}.attn.proj'] = proj_config
        quant_config[f'blocks.{i}.mlp.fc1'] = fc1_config
        quant_config[f'blocks.{i}.mlp.fc2'] = fc2_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model, nbits, group_size):
    # plot_weight_distribution(model, 32)
    
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=32)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=32)
    q4_16_config = BaseQuantizeConfig(nbits=4, group_size=16)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)
    q8_32_config = BaseQuantizeConfig(nbits=8, group_size=32)
    q8_128_config = BaseQuantizeConfig(nbits=8, group_size=128)
    
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_128_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_128_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_128_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_128_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_128_config
    return quant_config