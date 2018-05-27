import random
from torchloop.util.nlp_utils import lineToTensor
import torch

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# TODO: choose training example with a batch_size 
def batch_random_training_example(all_categories, category_lines):
    pass
