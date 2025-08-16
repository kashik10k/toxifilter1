import random

# Character substitutions
SUBS = {
    'a': ['@', '4'],
    'b': ['8'],
    'c': ['(', '{'],
    'e': ['3'],
    'g': ['9'],
    'i': ['1', '!', '|'],
    'l': ['1', '|'],
    'o': ['0'],
    's': ['$', '5'],
    't': ['7'],
    'u': ['v', 'ü'],
    'z': ['2'],
}

# Add common separators for break-ups
BREAKERS = ['', '.', '*', '-', '_']

def obfuscate_word(word, max_variants=5):
    """
    Generate multiple obfuscated versions of a toxic word.
    """
    variants = set()
    word = word.lower()

    while len(variants) < max_variants:
        new_word = ''
        for char in word:
            if char in SUBS:
                choice = random.choice([char] + SUBS[char])
            else:
                choice = char
            new_word += choice

        # Insert break characters randomly
        for breaker in BREAKERS:
            obfuscated = breaker.join(new_word)
            variants.add(obfuscated)

    return list(variants)

def augment_toxic_list(toxic_words, max_variants=5):
    """
    For a list of toxic words, generate obfuscated versions for each.
    """
    result = []
    for word in toxic_words:
        obfuscated = obfuscate_word(word, max_variants)
        result.extend(obfuscated)
    return result
