>>> import nltk
>>> nltk.download()
>>>
>>>
>>> noise_list = ["is", "as", "this", "...."]
>>> def _remove_noise(input_text):
...     words = input_text.split()
...     noise_free_words = [word for word in words if word not in noise_list]
...     noise_free_text = " ".join(noise_free_words)
...     return noise_free_text
...
>>> _remove_noise("this is a sample text")
'a sample text'
>>>
>>>
>>> import re
>>> def _remove_regex(input_text, regex_pattern):
...     urls = re.finditer(regex_pattern, input_text)
...     for i in urls:
...             input_text = re.sub(i.group().strip(), '', input_text)
...     return input_text
...
>>> regex_pattern = "#[\w]*"
>>>
>>> _remove_regex("remove this #hashtag from analytics fullarray", regex_pattern)
'remove this  from analytics fullarray'
>>>
>>>
>>> from nltk.stem.wordnet import WordNetLemmatizer
>>> lem = WordNetLemmatizer()
>>>
>>>
>>> from nltk.stem.porter import PorterStemmer
>>> stem = PorterStemmer()
>>>
>>> word = "multiplying"
>>> lem.lemmatize(word, "v")
'multiply'
>>> stem.stem(word)
'multipli'
>>> stem.stem(word)
'multipli'
>>> lem.lemmatize(word, "v")
'multiply'
>>>
>>> lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm":"awesome", "luv
":"love"}
>>>
>>> def _lookup_words(input_text):
...     words = input_text.split()
...     new_words = []
...     for word in words:
...             if word.lower() in lookup_dict:
...                     word = lookup_dict[word.lower()]
...             new_words.append(word)
...             new_text = " ".join(new_words)
...             return new_text
...
>>> _lookup_words("RT this is a retweeted tweet by fullarray")
'Retweet'
>>>
>>> exit()