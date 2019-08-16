from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def nlp_catout_train(X_train, y_train, max_len):
    """
    This function is to reformat the feature and target vectors (categorical data)to fit with sequencial model requirements (nlp task).
    The input are a matrix of features and a list of target.
    The output will change the features into the tensor form and the target are changed into a one-hot encoding.
    """
    #### X::
	# word tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(set(tokenizer.word_index))+1
        
	# padding and create feature tensors
    X_train_tensor = pad_sequences(tokenizer.texts_to_sequences(X_train.copy()), maxlen=max_len)           
    print('Shape of data tensor - train:', X_train_tensor.shape)
     
	#### y::
    # convert target to int class
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train.copy())
    
    y_train_encode = label_encoder.transform(y_train.copy())
    
    print('y before encoded:\n{}\n'.format(y_train[0:10]))
    print('y after encoded:\n{}\n'.format(y_train_encode[0:10]))
		
		# mapping class
    mapping_label = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))    
    print(mapping_label,'\n')
    print('n words: {}'.format(len(tokenizer.word_index)))
		
		# create one-hot encoding
    y_train_onehot = to_categorical([i for i in y_train_encode])

    return X_train_tensor, y_train_onehot, tokenizer, vocab_size, max_len




def nlp_catout_test(X_test, y_test, tokenizer, max_len):
    #### X::
	# word tokenizer
       
	# padding and create feature tensors
    X_test_tensor = pad_sequences(tokenizer.texts_to_sequences(X_test.copy()), maxlen=max_len)        
    
    print('Shape of data tensor - train test:', X_test_tensor.shape)
    
    
	#### y::
    # convert target to int class
    y_test_onehot = []
    if len(y_test) > 0:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test.copy())
        y_test_encode = label_encoder.transform(y_test.copy())
        
        print('y before encoded:\n{}\n'.format(y_test[0:10]))
        print('y after encoded:\n{}\n'.format(y_test_encode[0:10]))
        
        # mapping class
        mapping_label = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        
        print(mapping_label,'\n')
        print('n words: {}'.format(len(tokenizer.word_index)))
    
        # create one-hot encoding
        y_test_onehot = to_categorical([i for i in y_test_encode])
    
    return X_test_tensor, y_test_onehot