import torch
from fire import Fire

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from peft import PeftModel, LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import (
    estimate_gradient,
    LoraGAContext,
    save_loraga_model_init,
    save_loraga_model_final,
)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from accelerate import Accelerator
from utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from data import DATASET_MAP
import wandb
import os


def main(lora_alpha=8, lora_rank=32, sample_size=128, seed=31):
    accelerator = Accelerator()
    model_id = "t5-base"
    model_type = "ConditionalGeneration"
    model_dtype = "bf16"
    dataset_name = "sst2"

    config = dict(
        model="t5-base",
        d=dataset_name,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
    )
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="offline",
            group="test",
            project="LoRA-GA in PEFT",
        )
    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )
    if accelerator.is_local_main_process:
        print(model)

    peft_config = LoraGAConfig(
        target_modules=find_all_linear_modules(model=model),
        lora_alpha=config["a"],
        r=config["r"],
        iters=config["s"] // 2,
    )

    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    if isinstance(train_set, list):
        temp_set = train_set[: peft_config.bsz * peft_config.iters]
    else:
        temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
    transform_dataset(
        model_type=model_type,
        dataset=temp_set,
        tokenizer=tokenizer,
        max_length=peft_config.max_length,
    )
    dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)

    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=False,
    )
    if accelerator.is_local_main_process:
        print(peft_config)
    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)

    save_dir = os.path.join("./snapshot", wandb_name)
    if accelerator.is_local_main_process:
        print(model)
        save_loraga_model_init(model=model, save_dir=save_dir)
    print("finish get_peft_model=================================================")

    model = train_text_to_text_model(
        run_name=os.path.join("peft_test", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        num_train_epochs=1,
        per_device_batch_size=32,
        real_batch_size=32 * accelerator.num_processes,
        bf16=(model_dtype == "bf16"),
        eval_epochs=1,
        early_stopping_patience=5,
        max_length=128,
        logging_steps=1,
        use_loraplus=False,
        loraplus_lr_ratio=None,                      
        learning_rate=1e-4,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        training_args=dict(
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
        ),
    )
    if accelerator.is_local_main_process:
        save_loraga_model_final(model=model, save_dir=save_dir)
        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        model = PeftModel.from_pretrained(model, save_dir)
        print(model)


if __name__ == "__main__":
    Fire(main)
