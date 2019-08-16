import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from num2words import num2words
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
from tqdm import tqdm


# load english stopword
stop_words = set(stopwords.words('english'))


# add specific stopwords
def add_stopwrd(word_list):
    """
    This function is to add additional stopwords.
    The input is a list of strings and the output will be a list of stopwords.
    """
    stop_words.update(word_list)
    return stop_words
 
    
# clean stop words
def stopwrd_clean(X_in):
    """
    This function is to elimiate english stopwords.
    The input is a list of sentences and the output will be a list of sentences without stopwords.
    """
    X = []
    for i in range(len(X_in)):
        temp = []
        [temp.append(w) for w in X_in[i].split() if w not in stop_words]
        X.append(' '.join(temp))
    X_in = X
    return X_in


# load english word neglation
contraction_dict = {"ain't": "is not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "'cause": "because",
                    "could've": "could have",
                    "couldn't": "could not",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'll": "he will",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is",
                    "I'd": "I would",
                    "I'd've": "I would have",
                    "I'll": "I will",
                    "I'll've": "I will have","I'm": "I am",
                    "I've": "I have",
                    "i'd": "i would",
                    "i'd've": "i would have",
                    "i'll": "i will",
                    "i'll've": "i will have",
                    "i'm": "i am",
                    "i've": "i have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have",
                    "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not",
                    "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she would",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "this's": "this is",
                    "that'd": "that would",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "here's": "here is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why is",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you would",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
                   }

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)
contractions, contractions_re = _get_contractions(contraction_dict)


# word normalization
def txt_norm(X_in, lower = True, newline = True, special_char = True, dup_whitespace = True, n2w = True,
             negation = True, apost = True, lemmatize = True, stopw = True):
    """
    This function is used to normalize text.
    The input is a list of sentences.
    The scope is as below;
        1. Stopwords
        2. Negation
        3. Lower case
        4. Newline
        5. Special characters
        6. White space
        7. Number in numeric and number in text
        8. Apostrophe
        9. lemmatize
    The output will be a list of text.
    """         
    # Change to lowercase    
    if lower:
        X_in = [i.lower() for i in X_in]

    # Remove \r
    if newline:
        X_in = [i.replace('\r',' ') for i in X_in]
    
    # Remove special characters
    # 1) Replace special characters except ' . - :
    # 2) Replace % as percent
    # 3) Replace /s-/s as whitespace
    # 4) Replace - as to
    # 5) Replace .00 :00 as whitespace
    # 6) Replace : . as whitespace  
    if special_char:
        X_in = [re.sub('[^a-zA-Z0-9\'\.\-\:]', ' ', i).replace('%', 'percent')\
                .replace(' - ', ' ').replace('-', ' to ').replace('.00','').replace(':00','')\
                .replace(':',' ').replace('.', ' ') for i in X_in]
    
    # Remove duplicate whitespace   
    if dup_whitespace:
        X_in = [re.sub('\s+', ' ', i).strip() for i in X_in]

    # Change number into words
    if n2w: 
        X = []
        for item in X_in:
            temp = []
            [temp.append(num2words(i).replace(' point zero','')) if i.isdigit() == True else temp.append(i) for i in item.split(' ')]
            X.append(' '.join(temp))
        X_in = X

    # Negation handling
    if negation:
        X_in = [replace_contractions(i) for i in X_in]

    # Lemmatization
    # : Not using stemming in this case since lemmatization will give more proper words (understandable result)
    if lemmatize:
        X_lem = []
        wordnet_lemmatizer = WordNetLemmatizer()
        for i in range(len(X_in)):
            temp = []
            [temp.append(wordnet_lemmatizer.lemmatize(w)) for w in X[i].split()]
            X_lem.append(' '.join(temp))
        X_in = X_lem
       
    # Remove '
    if apost:
        X_in = [re.sub("'\s[a-z]\s|'[a-z]|'\s", ' ', i) for i in X_in]

    # Remove stopwords
    if stopw:
        X_in = stopwrd_clean(X_in)
    return X_in


def unpack(seq):
    if isinstance(seq, (list, tuple, set)):
        yield from (x for y in seq for x in unpack(y))
    elif isinstance(seq, dict):
        yield from (x for item in seq.items() for y in item for x in unpack(y))
    else:
        yield seq
        

#POS and shallow parsing (np = noun phrase)
def shallow_np(X):
    """
    This function is to extract "noun phrase" from a list of comments (X)
    """
    X_ = txt_norm(X, lower=False, stopw=False) #need to keep the sentence structure to extract the np
    
    #noun phrase extract
    nphrase = []
    n = 0
    for n in tqdm(range(len(X_))):
        temp = TextBlob(X_[n],np_extractor=ConllExtractor()).noun_phrases.lower()
        nphrase.append(temp)
        nphrase = list(unpack(nphrase))
        
    temp = []    
    for i in range(len(nphrase)):
        temp.append(nphrase[i])
    nphrase2 = txt_norm(temp) #standardize words
    nphrase3 = [word for word in nphrase2 if word not in ''] #eliminate the element which is removed during text normalization process (e.g., stopword)
    
    return X_, nphrase3