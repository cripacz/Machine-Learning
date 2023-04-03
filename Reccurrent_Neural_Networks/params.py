EMBEDDING_DIM = input("Embedding dimension: ")
VOCAB_SIZE = input("Vocabulary size: ")
MAXLEN = input("Maximum length of sentence: ")
dropout = input ("Dropout: ")
LearningRate = input ('Learning rate: ')

print("Hyper-parameters chosen:\n Embedding dimension: ", EMBEDDING_DIM,
      "\n Vocabulary size: ", VOCAB_SIZE, "\n Maximum length of sentence: ", MAXLEN, 
      '\n Dropout: ', dropout, '\n Learning rate: ', LearningRate)