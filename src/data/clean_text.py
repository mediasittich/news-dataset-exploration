import re
import string
import emoji

def remove_numbers(txt):
    return re.sub(r"\d+", '', str(txt))

def remove_hash(txt):
    return re.sub(r"#", '', txt)

def remove_urls_email(txt):
    clean_str = re.sub(r"\S*@\S*\s?", '', txt)
    clean_str = re.sub(r"https?://[^\s\n\r]+", '', clean_str)
    return clean_str

def remove_emojis(txt):
    return emoji.replace_emoji(txt, replace='')

def remove_punctuation(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_whitespace(txt):
    return ' '.join(re.sub('[\s+]', ' ', txt).split())

def clean_txt(txt):
    clean_txt = remove_numbers(txt)
    clean_txt = remove_hash(clean_txt)
    clean_txt = remove_urls_email(clean_txt)
    clean_txt = remove_emojis(clean_txt)
    clean_txt = remove_punctuation(clean_txt)
    clean_txt = remove_whitespace(clean_txt)
    return clean_txt

def calculate_word_count(txt):
    return len(txt.split(' '))

