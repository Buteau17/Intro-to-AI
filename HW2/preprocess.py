import nltk
import string
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer




def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english') 
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    def remove_punctuation (text: str) -> str:
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token for token in tokens if token != '&amp;']
        filtered_tokens = [token for token in tokens if token  not in string.punctuation]
        preprocessed_text = ' '.join(filtered_tokens)

        return preprocessed_text


    
    

    preprocessed_text = remove_punctuation(text)
    




    # End your code

    return preprocessed_text

# if __name__ == '__main__':
#     text = 'Here is a dog'
#     print(text)
#     print('output:', preprocessing_function(text))