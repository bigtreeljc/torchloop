from pydoc import locate

def for_name(cls_str):
    return locate(cls_str)

def batched_forname(*cls):
    ret = []
    for cls_ in cls:
        ret.append(locate(cls_))
    return tuple(ret)
