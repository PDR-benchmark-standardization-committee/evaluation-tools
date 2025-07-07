#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_traj_with_yaw(ax, df, label, color='black', yaw_color='limegreen', yaw_interval=1, skip_set_options=False):
    """
    traj1 : royalblue  / yaw1 : skyblue
    traj2 : darkorange / yaw2 : gold
    """
    if not skip_set_options:
        ax.set_title("Trajectories with Yaw Angles")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        ax.grid(True)

    ax.plot(df['x'], df['y'], label=label, color=color)
    ax = plot_yaw(ax, df, yaw_color, yaw_interval)

    return ax


def plot_yaw(ax, df, color, interval):
    """
    df : [timestamp, x, y, yaw]
    interval : 表示頻度
    """
    indices = np.arange(0, len(df), interval)
    x = df['x'].iloc[indices]
    y = df['y'].iloc[indices]
    yaw = df['yaw'].iloc[indices]

    # 矢印の成分（単位ベクトル）
    u = np.cos(yaw)
    v = np.sin(yaw)

    ax.quiver(x, y, u, v, color=color, angles='xy',
              scale_units='xy', scale=5, width=0.003)

    return ax
