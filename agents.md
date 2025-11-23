# Python programming guidance
* Prioritize clarity rather than performance, unless specified.
* Write code concisely. Don't generate files if it's logically required.
* Make comments only if it's nontrivial, requires explanation for later maintenance.
* Check always if `final.py` runs correctly, if crashes, debug it.
* Don't try to list all elements of dataset, or all lines from dataset. Read only a few lines of it.
* Don't give names too lengthy. Use meaningful names that describe the purpose of the variable or function.
* Try to run script always after you made. Debug if crashes.
* Use GPu `mps` not `cuda`.

# Repository explanation
* The directory `runs` includes `.pt` which are pretrained checkpoint for models.
* All visualization made should be placed under `visualizations`.
* The directory `scripts` gathers python scripts which are runnable on its own.
* training dataset is under `20GBprocessed`

## final.py
It contains runnable final evaluation script of model.

# Programming style
It should strictly follow OOP, hence class always expose the essential methods hiding else.
Don't separate classes too often, only do it when it's necessary.
Non-trivial inheritance or polymorphism is discouraged.

# Data preprocessing
* Use relative paths for writing dataset, retrieving it. Use absolute path if it is mandatory.
