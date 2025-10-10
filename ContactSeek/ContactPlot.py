# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ===================== plot1: plot contact-region [heatmap] =====================
def plot_contact_region_correlation(corr_matrix, kept_indices, contact_regions, 
                                  region_start=1, region_end=None,
                                  protein_name="Cas", save_path=None,
                                  figsize=(12, 10)):
    """
    绘制特定区域的correlation矩阵，并标注contact regions
    
    Parameters:
    -----------
    corr_matrix : numpy.ndarray
        完整的相关性矩阵
    kept_indices : numpy.ndarray
        保留的残基索引
    contact_regions : list
        contact region列表
    region_start : int
        区域起始位置（1-based）
    region_end : int
        区域结束位置（1-based），如果为None则显示全部
    protein_name : str
        蛋白名称
    save_path : str
        保存路径
    figsize : tuple
        图形大小
    """
    if save_path is None:
        save_path = f'{protein_name.lower()}_contact_regions_correlation.png'
    
    # 如果没有指定结束位置，显示全部
    if region_end is None:
        region_end = kept_indices[-1] + 1
        
    # 转换为0-based索引
    region_start_0 = region_start - 1
    region_end_0 = region_end - 1
    
    # 找到区域内的kept_indices位置
    region_mask = (kept_indices >= region_start_0) & (kept_indices <= region_end_0)
    region_positions = np.where(region_mask)[0]
    region_residues = kept_indices[region_mask]
    
    print(f"\n{protein_name} - Region {region_start}-{region_end} analysis:")
    print(f"Found {len(region_residues)} residues in kept_indices")
    
    if len(region_residues) == 0:
        print(f"Warning: No kept residues found in region {region_start}-{region_end}")
        return None
    
    # 提取子相关性矩阵
    sub_corr = corr_matrix[np.ix_(region_positions, region_positions)]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # ========== 绘制相关性矩阵 ==========
    im = ax.imshow(sub_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 找出该区域内的contact regions
    region_contact_regions = []
    
    for cr in contact_regions:
        cr_positions = cr['positions']
        # 检查contact region是否在目标区域内
        if np.any((cr_positions >= region_start_0) & (cr_positions <= region_end_0)):
            # 找到contact region中在区域内的残基
            cr_in_region = []
            for pos in cr_positions:
                if pos in region_residues:
                    idx_in_region = np.where(region_residues == pos)[0][0]
                    cr_in_region.append(idx_in_region)
            
            if cr_in_region:
                region_contact_regions.append({
                    'indices': cr_in_region,
                    'id': cr['contact_region_id'],
                    'type': cr['type'],
                    'avg_corr': cr['avg_correlation'],
                    'size': cr['size'],
                    'original_positions': cr['positions']
                })
    
    # 绘制contact region边界
    colors = plt.cm.tab20(np.linspace(0, 1, len(region_contact_regions)))
    
    for i, cr_info in enumerate(region_contact_regions):
        indices = cr_info['indices']
        if len(indices) > 0:
            # 绘制矩形边界
            min_idx = min(indices)
            max_idx = max(indices)
            
            # 主对角线区域
            rect = Rectangle((min_idx - 0.5, min_idx - 0.5), 
                           max_idx - min_idx + 1, 
                           max_idx - min_idx + 1,
                           linewidth=2.5, edgecolor=colors[i], 
                           facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # 添加标签 - 简化版本，只显示ID
            label_text = f"CR{cr_info['id']}"
            
            # 标签位置 - 放在矩形框的右上角外侧
            ax.text(max_idx + 1, min_idx - 0.5, 
                    label_text, 
                    fontsize=10, fontweight='bold',
                    color=colors[i], ha='left', va='top')
    
    # 设置刻度
    # 根据区域大小动态调整刻度数量
    n_ticks = min(20, len(region_residues))
    tick_indices = np.linspace(0, len(region_residues)-1, n_ticks, dtype=int)
    
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([region_residues[i] + 1 for i in tick_indices], rotation=90, fontsize=8)
    ax.set_yticklabels([region_residues[i] + 1 for i in tick_indices], fontsize=8)
    
    ax.set_title(f'{protein_name} Contact Regions: {region_start}-{region_end}', fontsize=14)
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Residue Position', fontsize=12)
    
    # 添加细网格
    ax.set_xticks(np.arange(len(region_residues) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(region_residues) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印找到的contact regions
    print(f"\nFound {len(region_contact_regions)} contact regions in this range:")
    for cr_info in sorted(region_contact_regions, key=lambda x: x['indices'][0]):
        orig_pos = cr_info['original_positions']
        range_str = f"{orig_pos[0]+1}-{orig_pos[-1]+1}" if len(orig_pos) > 1 else f"{orig_pos[0]+1}"
        print(f"  CR{cr_info['id']}: {range_str} (size={cr_info['size']}, avg_corr={cr_info['avg_corr']:.3f})")
    
    return fig