import torch
import typing as tp
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from logTrainer import LogTrainer
import logging


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


def find_all_linear_modules(model) -> tp.List[str]:
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


def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
    is_list_input = isinstance(example["x"], list)
    # Combine text and add EOS token
    combined_text = (
        [
            x + " " + y + tokenizer.eos_token
            for (x, y) in zip(example["x"], example["y"])
        ]
        if is_list_input
        else example["x"] + " " + example["y"] + tokenizer.eos_token
    )
    # Tokenize combined text
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    # Calculate input text length in tokens
    input_text_length = (
        [
            len(tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
            for i in range(len(example["x"]))
        ]
        if is_list_input
        else len(tokenizer(example["x"], return_tensors="pt")["input_ids"][0])
    )
    if input_text_length[0] >= max_length:
        log.warning(
            f"Input text length >= max_length: {input_text_length} >= {max_length}. "
            "Consider increasing max_length to avoid truncation."
        )
    # Create labels
    labels = encodings["input_ids"].clone()
    if is_list_input:
        for i, l in enumerate(input_text_length):
            labels[i, :l] = -100
    else:
        labels[0, :input_text_length] = -100
    if ignore_masked_token:
        labels[encodings["attention_mask"] == 0] = -100
    # Update example dictionary
    results = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        # "input_text_length": input_text_length,
    }

    return results


def SeqToSeqEncode(example, tokenizer, max_length=None, ignore_masked_token=False):
    inputs = tokenizer(
        example["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
        "decoder_attention_mask": outputs["attention_mask"],
    }

    if ignore_masked_token:
        results["labels"][outputs["attention_mask"] == 0] = -100

    return results


def preprocess_dataset(
    dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
) -> Dataset:
    if isinstance(dataset, list) and isinstance(dataset[0], tuple):
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        dataset = Dataset.from_dict(
            {k: [dic[k] for dic in dataset] for k in dataset[0]}
        )
    elif isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    elif isinstance(dataset, Dataset):
        pass
    else:
        raise ValueError("Wrong format")
    return dataset


def initialize_text_to_text_model(
    model_name: str,
    model_type: str,
    dtype: str,
    tokenizer: str = None,
    flash_attention: bool = False,
):
    assert model_type in ["CausalLM", "ConditionalGeneration"]
    auto_model_class = (
        AutoModelForCausalLM if model_type == "CausalLM" else AutoModelForSeq2SeqLM
    )
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        log.info("Using flash attention 2")
        model_config["attn_implementation"] = "flash_attention_2"
    match dtype:
        case "fp32":
            model_config["torch_dtype"] = torch.float32
        case "bf16":
            model_config["torch_dtype"] = torch.bfloat16
        case "int8":
            quant_8bit_config = dict(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                # llm_int8_has_fp16_weight=False
            )
            model_config["quantization_config"] = quant_8bit_config
        case "nf4":
            quant_4bit_config = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_config["quantization_config"] = quant_4bit_config
        case _:
            raise ValueError("Wrong dtype")
    model = auto_model_class.from_pretrained(**model_config)
    if tokenizer:
        log.info(f"Using custom tokenizer {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def transform_dataset(model_type, tokenizer, dataset, max_length):
    if model_type == "CausalLM":
        dataset.set_transform(lambda x: causalLMEncode(x, tokenizer, max_length))
    elif model_type == "ConditionalGeneration":
        dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, max_length))
    else:
        raise ValueError("Wrong model type")
    return dataset


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    **kwargs,
) -> torch.nn.Module:
    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // (
        per_device_batch_size * kwargs.get("num_process", 1)
    )
    train_dataset, valid_dataset = transform_dataset(
        model_type, tokenizer, train_dataset, max_length
    ), transform_dataset(model_type, tokenizer, valid_dataset, max_length)

    eval_steps = (
        int(len(train_dataset) * kwargs.get("eval_epochs", 1)) // real_batch_size
    )
    TrainingArgumentsClass = Seq2SeqTrainingArguments
    TrainerClass = LogTrainer
    output_dir = f"./results/{run_name}/{kwargs.get('seed')}"
    training_args = TrainingArgumentsClass(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("num_train_epochs", 1),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",
        logging_steps=kwargs.get("logging_steps", 10),
        bf16=kwargs.get("bf16", False),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        optim=kwargs.get("optim", "adamw_torch"),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=kwargs.get("load_best_model_at_end", True),
        metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
        greater_is_better=kwargs.get("greater_is_better", False),
        do_eval=True,
        learning_rate=kwargs.get("learning_rate", 5e-5),
        remove_unused_columns=False,  # tokenize the dataset on the fly
        # eval_accumulation_steps=kwargs.get("eval_accumulation_steps", real_batch_size),
        label_names=["labels"],
        seed=kwargs.get("seed", 42),
        ddp_find_unused_parameters=False,
        **kwargs.get("training_args", {}),
    )
    """
    eval_accumulation_steps (int, optional) — Number of predictions steps to accumulate the output tensors for,
    before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/NPU/TPU before being moved to the CPU 
    (faster but requires more memory).
    
    if you want to specify compute_metrics for TrainingAguments, you can (should) specify preprocess_logits_for_metrics for Trainer to to avoid
    `cuda out of memory`
    """

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=None,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=kwargs.get("early_stopping_patience", 1)
            ),
        ],
    )

    trainer.train()

    return model
