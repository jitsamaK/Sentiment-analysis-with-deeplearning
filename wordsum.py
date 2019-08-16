import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


def bow(x):
    """
    This function is to create a bag of word table given the comment as list
    """
    word_f = Counter(x) #bag of words
    word_f_pd = pd.DataFrame.from_dict(dict(word_f), orient='index', columns=['freq']).sort_values(by=['freq'], axis=0, ascending=False)
    word_f_pd = word_f_pd.reset_index(drop=False)
    
    return word_f, word_f_pd
    

_novalue = object()

def wordc(word_f=_novalue, clr=_novalue, title=_novalue):
    """
    This function is to create a wordcloud where;
        word_f = bag of word (dictionay)
        clr = matplotlib color scheme for wordcloud
        title = string for wordcloud name
    """
    if clr is _novalue:
        clr = matplotlib.cm.hot
        
    if title is _novalue:
        title = ''
        
    wordcloud = WordCloud(width=900,height=500, random_state=2304, colormap=clr,
                          relative_scaling=1,normalize_plurals=False).fit_words(word_f)
    plt.figure(figsize=(15,8))
    plt.title(str(title), fontsize=20)
    plt.imshow(wordcloud)
    plt.axis('off')
    return plt