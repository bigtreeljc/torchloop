import random
from torchloop.util import nlp_utils
import torch

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = nlp_utils.lineToTensor(line)
    return category, line, category_tensor, line_tensor

