# TODO

1. `seed_workspaces/nemo_mas_reasoner/model/adapter.yaml`
   - Do not stack LoRA on top of LoRA.
   - Each SFT run should train directly from the base model instead of starting from `seed_adapter_path`.

2. LoRA search/config
   - Set LoRA rank directly to `32`.
   - Try more target modules beyond attention-only:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
]
```
