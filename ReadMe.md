# Project Setup Guide

This guide explains how to create a Conda environment with Python 3.10, install dependencies from `requirements.txt`, and run the project.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

---

## 1. Download the Repository

```bash
git clone https://github.com/dapowan/MMBox.git
cd MMBox

```

## 2. Create & Activate Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n env_mmb python=3.10 -y

# Activate the environment
conda activate env_mmb

```
## 3. Install Dependencies

```bash
# Create a new conda environment with Python 3.10
pip install -r requirements.txt
```
如果上面的不行，就用下面，其实就安装了一个ms-swift
```bash
pip install ms-swift

```

## 4. Run the Project
```bash
# 赋权
chmod +x mmb_sft.sh
# 运行
./run.sh
```