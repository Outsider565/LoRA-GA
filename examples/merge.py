from fire import Fire
from peft import PeftModel
from utils import initialize_text_to_text_model
import os
import torch
from peft.tuners.lora.bnb import (
    Linear8bitLt as LoraLinear8bitLt,
    Linear4bit as LoraLinear4bit,
)


def get_float_weight(model: torch.nn.Module):
    model: torch.nn.Linear

    device = model.weight.device
    in_features = model.in_features
    with torch.no_grad():
        I = torch.eye(in_features).to(device)
        w = model(I)
        if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = model.weight.requires_grad
    return w


def replace_A_with_Linear(model: torch.nn.Module, target):
    for name, module in model.named_children():
        if isinstance(module, target):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(module).data
                if bias:
                    new_module.bias.data = (
                        module.bias if module.bias is not None else None
                    )
            setattr(model, name, new_module)

        else:
            replace_A_with_Linear(module, target)


def dequantize(model, dtype):
    if dtype == "int8":
        target = LoraLinear8bitLt
    elif dtype = "nf4":
        target == LoraLinear4bit
    replace_A_with_Linear(model=model, target=target)


def merge(
    checkpoint: str,
    dtype: str,
    model_name="meta-llama/Llama-2-7b-hf",
    model_type="CausalLM",
    merge_suffix="merged_checkpoint",
):
    # model_name = "t5-base"
    # model_type = "ConditionalGeneration"
    # model_name = "meta-llama/Llama-2-7b-hf"
    # model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, dtype="bf16"
    )

    if dtype in ["nf4"]:
        float_attr_list = list(model.__dict__.keys())
        float_llama_config_attr_list = list(model.config.__dict__.keys())
        del model
        torch.cuda.empty_cache()
        model, tokenizer = initialize_text_to_text_model(
            model_name, model_type, dtype=dtype
        )
        print(f"dtype of model is {dtype}, so dequantize model")
        print("dequantize======================================")
        print(model)
        from utils import show_gpu_and_cpu_memory

        show_gpu_and_cpu_memory()
        model = model.dequantize()
        quant_attr_list = list(model.__dict__.keys())
        quant_llama_config_attr_list = list(model.config.__dict__.keys())
        for attr in quant_attr_list:
            if attr not in float_attr_list:
                delattr(model, attr)
        for attr in quant_llama_config_attr_list:
            if attr not in float_llama_config_attr_list:
                delattr(model.config, attr)
        model = model.bfloat16()
        model = model.to("cpu")
        print("finish dequnatize=======================================")
        print(model)
    elif dtype in ["int8"]:
        dequantize(model, dtype)
        model = model.bfloat16()
        model = model.to("cpu")
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(checkpoint, merge_suffix))
    tokenizer.save_pretrained(os.path.join(checkpoint, merge_suffix))


if __name__ == "__main__":
    Fire(merge)
