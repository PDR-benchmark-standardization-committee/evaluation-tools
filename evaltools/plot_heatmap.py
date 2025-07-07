import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2

def main(pgm_path, map_origin, gt_csvs, evaluation_csvs, output_dir, title_txt='traj', grid_size=0.25):
    pgm_map_array = cv2.imread(pgm_path)
    gt_data = pd.concat([pd.read_csv(filename) for filename in gt_csvs], ignore_index=True)
    
    # Plot heatmap overlay
    plot_heatmap_overlay(pgm_map_array, gt_data, evaluation_csvs, map_origin, output_dir, 'heatmap_overlay', grid_size)

def plot_heatmap_overlay(pgm_map_array, gt_data, evaluation_csvs, map_origin, output_filename, title_txt, grid_size):
    eval_data = pd.concat([pd.read_csv(filename) for filename in evaluation_csvs], ignore_index=True)

    # Extract necessary columns from evaluation data
    ce_values = eval_data[eval_data['type'] == 'ce'][['timestamp', 'value']]

    # Merge ground truth data with evaluation data on timestamp
    merged_data = pd.merge_asof(gt_data.sort_values('timestamp'),
                                ce_values.sort_values('timestamp'),
                                on='timestamp', direction='nearest')

    # Calculate grid limits based on the merged data
    x_min, x_max = merged_data['x'].min(), merged_data['x'].max()
    y_min, y_max = merged_data['y'].min(), merged_data['y'].max()

    # Create grid
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Assign grid indices to each point
    merged_data['x_bin'] = np.digitize(merged_data['x'], x_bins) - 1
    merged_data['y_bin'] = np.digitize(merged_data['y'], y_bins) - 1

    # Create an empty heatmap with NaN as default values
    heatmap = pd.DataFrame(index=np.arange(len(x_bins)-1), columns=np.arange(len(y_bins)-1), dtype=float).fillna(np.nan)

    # Fill the heatmap with mean values where data exists
    mean_values = merged_data.groupby(['x_bin', 'y_bin'])['value'].mean()
    for (x_bin, y_bin), value in mean_values.items():
        heatmap.at[x_bin, y_bin] = value

    # Plot the heatmap overlay
    fig, ax = plt.subplots(figsize=(30, 20))
    map_x_min = map_origin[0]
    map_y_min = map_origin[1]
    map_x_max = map_x_min + pgm_map_array.shape[1] / 20.0  # 20 pixels per meter
    map_y_max = map_y_min + pgm_map_array.shape[0] / 20.0

    ax.imshow(pgm_map_array, extent=[map_x_min, map_x_max, map_y_min, map_y_max], alpha=1.0)
    cax = ax.imshow(heatmap.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap='jet', alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('CE Heatmap Overlay')
    ax.set_aspect('equal')
    fig.colorbar(cax, ax=ax, label='CE (Mean)')
    fig.savefig(F'{output_filename}.png', dpi=200)


def transform_opencv(x, y, mapY_len, O_map=(0, 0)):
    return ((x - O_map[0])*20).astype(int), mapY_len - ((y - O_map[1])*20).astype(int)


def main_cl():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pgm_path', '-p', help='Path to the PGM file')
    parser.add_argument('--map_origin', '-m', nargs=2, type=float, help='Origin of the map (O_map). --map-origin {x(m)} {y(m)}')
    parser.add_argument('--gt_csvs', '-g', type=str, nargs='+', help='Ground Truth CSV files')
    parser.add_argument('--evaluation_csvs', '-e', nargs='+', help='Evaluation result CSV files')
    parser.add_argument('--title_txt', '-t', default='traj', help='Title of the output plot')
    parser.add_argument('--output_filename', '-o', default='heatmap_overlay.png', help='Filename to output results')
    parser.add_argument('--grid_size', '-gs', type=float, default=0.25, help='Grid size in meters')

    args = parser.parse_args()
    main(args.pgm_path, tuple(args.map_origin), args.gt_csvs, args.evaluation_csvs, args.output_filename, args.title_txt, args.grid_size)


if __name__ == '__main__':
    main_cl()