# Directory for helpers modules

## prompter.py

Prompter class, a template manager.

`from utils.prompter import Prompter`

## monkeypatches
Module which contains the monkeypatches (flash attention and xformers) to replace the default attention for LLaMA. 
Other models will be added in the future. 

`from utils.monkeypatches import [func_name]`

## callbacks.py

Helpers to support streaming generate output.

`from utils.callbacks import Iteratorize, Stream`
