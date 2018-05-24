def all_ascii_characters():
    return string.ascii_letters + " .,;':"

def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
    )


