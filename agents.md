# Python programming guidance
* Prioritize clarity rather than performance, unless specified.
* Write code concisely. Don't generate files if it's logically required.
* Make comments only if it's nontrivial, requires explanation for later maintenance.

# Repository explanation
* The directory `runs` includes `.pt` which are pretrained checkpoint for models.
* All visualization made should be placed under `visualizations`.

## final.py
It contains runnable final evaluation script of model.

# Programming style
It should strictly follow OOP, hence class always expose the essential methods hiding else.
Don't separate classes too often, only do it when it's necessary.
Non-trivial inheritance or polymorphism is discouraged.
