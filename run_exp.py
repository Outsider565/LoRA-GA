from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
)
import json
import math
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        if init_config.get("scale", "") == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        U, S, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
        V = V.T
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0 : 2 * lora_r : 2]
            A = V[1 : 2 * lora_r : 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())

    with torch.no_grad():
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config.dtype == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config.dtype == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )
        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
            module.weight.data.device
        )
        scaling_factor = module.scaling["default"]
        offset *= scaling_factor
        if "norm_clip" in init_config and init_config.norm_clip:
            # for numerical stability, offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                module.lora_A.default.weight.data *= ratio**0.5
                module.lora_B.default.weight.data *= ratio**0.5
                log.warning(f"Clipping offset by {ratio}")
        try:
            module.weight.data -= offset
        except:
            breakpoint()


def reinit_lora(model, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, **kwargs)

    return model


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    num = 0
    for batch in tqdm(dataloader, desc="Estimating gradient"):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    wandb.init(
        entity="atcs_project",
        project=cfg.wandb.project,
        name=name,
        config=config,
    )
    train_set, val_set, _ = dataset_func()
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    )
    additional_kwargs = {}
    if use_peft and cfg.init.mode == "gradient":
        if isinstance(train_set, list):
            temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
        else:
            temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
        transform_dataset(
            model_type=model_type,
            dataset=temp_set,
            tokenizer=tokenizer,
            max_length=cfg.init.max_length,
        )
        named_grads = estimate_gradient(model, temp_set, cfg.init.bsz)
        additional_kwargs["named_grads"] = named_grads

    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        reinit_lora(model, cfg.init, **additional_kwargs)
        if train_embeddings:
            model.lm_head.weight.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
        save_dir = os.path.join(
            "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "orig_checkpoint"
        )
        model.save_pretrained(save_dir)
        adapter_config = json.load(open(os.path.join(save_dir, "adapter_config.json")))
        adapter_config["lora_alpha"] = -adapter_config["lora_alpha"]
        json.dump(
            adapter_config, open(os.path.join(save_dir, "adapter_config.json"), "w")
        )
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }
    log.info(rate)
    # log rate into wandb summary
    wandb.summary.update(rate)
    training_loop = train_text_to_text_model
    model = training_loop(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        model,
        tokenizer,
        model_type,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        # deepspeed=(
        #     "z3_offload_all_bf16.json" if cfg.peft == False else None
        # ),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.seed,
    )
    save_dir = os.path.join(
        "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "merged_checkpoint"
    )
    if not use_peft:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        # merge_llama(os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}"))
        pass
    log.info(f"Saving model to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    run_exp()
