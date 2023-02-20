from collections import Counter


def get_common_chars_number(text1, text2):
    count_a = Counter(text1)
    count_b = Counter(text2)

    common_keys = set(count_a.keys()).intersection(count_b.keys())
    return sum(min(count_a[key], count_b[key]) for key in common_keys)


def get_common_chars_percent(text1, text2):
    return get_common_chars_number(text1, text2) / min(len(text1), len(text2))


def get_correct_placed_chars_percent(text1, text2):
    return get_correct_placed_chars_number(text1, text2) / min(len(text1), len(text2))


def get_correct_placed_chars_number(text1, text2):
    return sum(x == y for x, y in zip(text1, text2))


def is_length_same(text1, text2):
    return len(text1) == len(text2)


def is_text_same(text1, text2):
    return text1 == text2
