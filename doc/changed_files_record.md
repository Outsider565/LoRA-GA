- [Changed files table](#changed-files-table)
  - [Modified files](#modified-files)
  - [New files](#new-files)
- [Changed files list](#changed-files-list)
- [peft\_Lora-GA function call stack](#peft_lora-ga-function-call-stack)

## Changed files table

Some files only have a few lines modified and are not listed in these two tables.

### Modified files

| file                           | changes                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| src/peft/mapping.py            | add map `"LORAGA"`: `LoraGAConfig`                                                                |
| src/peft/peft_model.py         | add map `PeftType.LORAGA`: `LoraGAModel`                                                          |
| src/peft/tuners/lora/bnb.py    | LoraLayer.\_\_init\_\_(self, base_layer) --> LoraLayer.\_\_init\_\_(self, base_layer, \*\*kwargs) |
| src/peft/tuners/lora/config.py | add class `LoraGAConfig(LoraConfig)`                                                              |
| src/peft/tuners/lora/layer.py: | modify `LoraLayer.update_layer` ,add `LoraLayer.lora_ga_init`                                     |
| src/peft/tuners/lora/model.py  | add class `LoraGAModel(LoraModel`                                                                 |

### New files

| file                                                                             | role                                                                                  |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/context.py                  | a context manager for model offload and gradient offload                              |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/model_offload.py            | a context manager for model offload                                                   |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/gradient_offload.py         | a context manager for gradient offload(lomo)                                          |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/forward_backward_offload.py | a hook context for offload in forward and backward                                    |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/graph_hook.py               | [pack,unpack] for torch.autograd.graph.saved_tensors_hooks in saved_tensor_offload.py |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/saved_tensor_offload.py     | a context manager for saved tensor offload in computing graph                         |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/resource_monitor.py         | show gpu cpu memory                                                                   |
| src/peft/utils/lora_ga_utils/offload_utils_for_quant/split.py                    | decide split how many blocks                                                          |

## Changed files list

1. docs/source/developer_guides/lora.md

2. examples/lora_ga_finetuning/README.md

3. examples/lora_ga_finetuning/data.py

4. examples/lora_ga_finetuning/float_llama2-7b_metamath.py

5. examples/lora_ga_finetuning/logTrainer.py

6. examples/lora_ga_finetuning/lora_plus.py

7. examples/lora_ga_finetuning/quant_llama-2-7b_metamath.py

8. examples/lora_ga_finetuning/utils.py

9. src/peft/\_\_init\_\_.py

10. src/peft/mapping.py

11. src/peft/peft_model.py

12. src/peft/tuners/\_\_init\_\_.py

13. src/peft/tuners/lora/\_\_init\_\_.py

14. src/peft/tuners/lora/bnb.py

15. src/peft/tuners/lora/config.py

16. src/peft/tuners/lora/layer.py

17. src/peft/tuners/lora/model.py

18. src/peft/utils/lora_ga_utils/\_\_init\_\_.py

19. src/peft/utils/lora_ga_utils/lora_ga_utils.py

20. src/peft/utils/lora_ga_utils/offload_utils_for_quant/\_\_init\_\_.py

21. src/peft/utils/lora_ga_utils/offload_utils_for_quant/context.py

22. src/peft/utils/lora_ga_utils/offload_utils_for_quant/forward_backward_offload.py

23. src/peft/utils/lora_ga_utils/offload_utils_for_quant/forward_hook.py

24. src/peft/utils/lora_ga_utils/offload_utils_for_quant/gradient_offload.py

25. src/peft/utils/lora_ga_utils/offload_utils_for_quant/graph_hook.py

26. src/peft/utils/lora_ga_utils/offload_utils_for_quant/model_offload.py

27. src/peft/utils/lora_ga_utils/offload_utils_for_quant/resource_monitor.py

28. src/peft/utils/lora_ga_utils/offload_utils_for_quant/saved_tensor_offload.py

29. src/peft/utils/lora_ga_utils/offload_utils_for_quant/split.py

30. src/peft/utils/peft_types.py

31. src/peft/utils/save_and_load.py

## peft_Lora-GA function call stack

File "/root/project/lora-exp/test_peft/experiment_float.py", line 60, in main()

File "/root/project/lora-exp/test_peft/experiment_float.py", line 54, in main get_peft_model(model=model, peft_config=peft_config, adapter_name="default")

File "/root/project/lora-exp/peft/src/peft/mapping.py", line 175, in get_peft_model return PeftModel(model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)

File "/root/project/lora-exp/peft/src/peft/peft_model.py", line 155, in \_\_init\_\_ self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/model.py", line 921, in \_\_init\_\_ super().\_\_init\_\_(model, config, adapter_name)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/model.py", line 139, in \_\_init\_\_ super().\_\_init\_\_(model, config, adapter_name)

File "/root/project/lora-exp/peft/src/peft/tuners/tuners_utils.py", line 175, in \_\_init\_\_ self.inject_adapter(self.model, adapter_name)

File "/root/project/lora-exp/peft/src/peft/tuners/tuners_utils.py", line 431, in inject_adapter self.\_create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/model.py", line 983, in \_create_and_replace new_module = self.\_create_new_module(lora_config, adapter_name, target, \*\*kwargs)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/model.py", line 340, in \_create_new_module new_module = dispatcher(target, adapter_name, lora_config=lora_config, \*\*kwargs)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/layer.py", line 1202, in dispatch_default new_module = Linear(target, adapter_name, \*\*kwargs)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/layer.py", line 505, in \_\_init\_\_ self.update_layer(

File "/root/project/lora-exp/peft/src/peft/tuners/lora/layer.py", line 131, in update_layer self.lora_ga_init(adapter_name, init_lora_weights)

File "/root/project/lora-exp/peft/src/peft/tuners/lora/layer.py", line 173, in lora_ga_init traceback.print_stack()