# AnomalyDAE-PyG 精简复现版

这是对 `haoyfan/AnomalyDAE` 的 PyTorch Geometric 精简复现，去掉了 TensorFlow 1.x 旧工程、图片展示、冗余脚本和固定数据目录，只保留图异常检测所需的核心部分：

1. `.mat` 数据读取；
2. AnomalyDAE 模型；
3. 双重重构损失；
4. 训练、打分和 AUROC/AUPRC 评估。

## 1. 环境安装

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

## 2. 数据格式

支持常见图异常检测 `.mat` 数据：

- 邻接矩阵键名会自动从 `Network / A / adj / Adj / network` 中查找；
- 节点特征键名会自动从 `Attributes / X / features / attrb / attr` 中查找；
- 标签键名会自动从 `Label / labels / y / gnd / GroundTruth / anomaly_label` 中查找。


## 3. 运行示例

```bash
python run.py \
  --dataset BlogCatalog \
  --epochs 100 \
  --emb_dim 128 \
  --hid_dim 128 \
  --alpha 0.7 \
  --eta 5 \
  --theta 40 \
  --lr 0.001 \
  --device cuda:0
```

也可以使用数据集预设参数：

```bash
python run.py --dataset BlogCatalog --preset blogcatalog --device cuda:0
python run.py --dataset Flickr --preset flickr --device cuda:0
python run.py --dataset ACM --preset acm --device cuda:0
```

## 4. 主要参数说明

- `alpha`：结构重构误差权重，属性重构误差权重为 `1 - alpha`；
- `theta`：邻接矩阵非零元素重构误差额外惩罚；
- `eta`：属性矩阵非零元素重构误差额外惩罚；
- `emb_dim`：结构编码器和属性编码器第一层隐向量维度；
- `hid_dim`：最终节点嵌入和属性嵌入维度；
- `dropout`：Dropout；
- `topk`：无标签时输出 Top-K 异常节点。

## 5. 输出

程序会输出：

- 每若干 epoch 的训练 loss；
- 如果 `.mat` 内含标签，输出 AUROC 和 AUPRC；
- 保存异常分数：`outputs/anomaly_scores.npy`；
- 保存异常排序：`outputs/ranking.csv`。

## 6. 文件结构

```text
anomalydae_pyg/
├── run.py
├── requirements.txt
├── README.md
├── src/
│   ├── data.py
│   ├── loss.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── data/
└── examples/
    └── run_blogcatalog.sh
```
