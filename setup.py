#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup


def load_requirements(filename='requirements.txt'):
    """
    Load dependencies from a requirements.txt file.
    """
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name="evaltools",
    version="0.1.0",
    description="",
    author="Takuri Suzuki",
    author_email="suzuki-iluneco3@aist.go.jp",
    license="TBD",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=load_requirements(),
    entry_points={
        'console_scripts': [
            'do_eval = evaltools.do_eval:main_cl',
            'do_eval_rel = evaltools.do_eval_rel:main_cl',
            'plot_ecdf_from_csv = evaltools.plot_ecdf_from_csv:main_cl',
            'show_result = evaltools.show_result:main_cl',
            'show_result_ble = evaltools.show_result_ble:main_cl',
            'extract_csv_from_pickle = evaltools.extract_csv_from_pickle:main_cl',
            'plot_traj = evaltools.plot_traj:main_cl',
            'plot_heatmap = evaltools.plot_heatmap:main_cl',
            'plot_timeline = evaltools.plot_timeline:main_cl'
        ]
    }
)
