"""
预训练模型规格配置

定义不同规模的模型配置，方便在 train_model.py 中切换。
"""

# 模型规格（dim, n_layers, n_heads, n_kv_heads）
MODEL_CONFIGS = {
    # 约 26M 参数
    "26M": dict(dim=512, n_layers=8, n_heads=8, n_kv_heads=4),
    # 约 215M 参数（默认训练配置）
    "215M": dict(dim=1024, n_layers=18, n_heads=16, n_kv_heads=8),
    # 约 500M 参数
    "500M": dict(dim=1536, n_layers=24, n_heads=16, n_kv_heads=8),
}

# 默认训练超参数
DEFAULT_TRAIN_CONFIG = {
    "epochs": 1,
    "batch_size": 64,
    "learning_rate": 2e-4,
    "accumulation_steps": 8,
    "grad_clip": 1.0,
    "warmup_iters": 0,
    "log_interval": 100,
    "save_interval": 1000,
    "dtype": "bfloat16",
    "num_workers": 8,
}
