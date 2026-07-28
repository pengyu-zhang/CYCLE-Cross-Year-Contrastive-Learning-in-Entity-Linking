"""Optimizer and LR schedule (BLINK-compatible parameter selection)."""

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Which BERT parameters are trained, keyed by BLINK's
# --type_optimization values. The published runs used all_encoder_layers,
# i.e. encoder layers only (embeddings frozen).
PATTERNS = {
    "additional_layers": ["additional"],
    "top_layer": ["additional", "bert_model.encoder.layer.11."],
    "top4_layers": [
        "additional",
        "bert_model.encoder.layer.11.",
        "encoder.layer.10.",
        "encoder.layer.9.",
        "encoder.layer.8",
    ],
    "all_encoder_layers": ["additional", "bert_model.encoder.layer"],
    "all": ["additional", "bert_model.encoder.layer", "bert_model.embeddings"],
}
NO_DECAY = ["bias", "gamma", "beta"]


def build_optimizer(biencoder, type_optimization, learning_rate,
                    extra_modules=None, extra_lr=None):
    """AdamW over the selected BERT parameters (weight-decay split as in
    BLINK), plus optional extra parameter groups (graph module, fusion)."""
    patterns = PATTERNS[type_optimization]
    with_decay, without_decay = [], []
    for name, param in biencoder.named_parameters():
        if any(t in name for t in patterns):
            (without_decay if any(t in name for t in NO_DECAY) else with_decay).append(param)
        else:
            param.requires_grad_(False)

    groups = [
        {"params": with_decay, "weight_decay": 0.01},
        {"params": without_decay, "weight_decay": 0.0},
    ]
    for module in extra_modules or []:
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "weight_decay": 0.0, "lr": extra_lr})
    return AdamW(groups, lr=learning_rate)


def build_scheduler(optimizer, num_train_steps, warmup_proportion):
    warmup = int(num_train_steps * warmup_proportion)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (num_train_steps - step) / max(1, num_train_steps - warmup))

    return LambdaLR(optimizer, lr_lambda)
