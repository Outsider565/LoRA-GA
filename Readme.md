# LoRA-GA

Official implementation of the paper "LoRA-GA: Low-Rank Adaptation with Gradient Approximation".

## How to run

We use hydra to manage the configurations. You can find the default configurations in `conf/` directory.

We have three config groups: `peft`, `dataset`, and `init`. You can use `+peft=xxx` to specify the `peft` config, `+dataset_name=xxx` to specify the `dataset` config, and `+init=xxx` to specify the `init` config. You can also use `++peft.xxx=xxx` to specify the sub-configs of `peft`. Learn more about hydra [here](https://hydra.cc/docs/intro).

There are two modes: single run (for one dataset) and multi run (for all datasets).


For single run, you can use command like this

```bash
python run_exp.py +peft=all ++peft.lora_relative_r=0.1 +dataset_name=sst2 +init=gaussian 
```

For multi run, you can use command like this

```bash
python run_exp.py -m +init=gradient ++peft.lora_r=8 +peft=all wandb.name="stable-gradient-64" ++init.weight="stable" peft.use_rslora=True ++init.stable_gamma=64
```

## Configurations

In order to run LoRA-GA, you need to specify the following configurations:
- init=gradient
- init.weight=stable
- peft.use_rslora=True
In this way, you enable the +SO and +GA parts of LoRA-GA.

If you want to run the default LoRA, you can use the following configurations:
- init=default

## How to download the datasets

The datasets are automatically downloaded when you run the code. If you want to download the datasets manually, you can edit `data.py` and use the following command:

```bash
python data.py
```