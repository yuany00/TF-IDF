import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords



def _create_frequency_table(text_string):
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

#here the document is a paragragh , the term is a word in a paragragh
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix):
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
        
    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold ):
    sentence_count = 0
    summary = ''
    value =[]
    '''for sent in sentenceValue.keys():
        value.append(sentenceValue[sent])

    value.sort(reverse= True)
    threshold= value[sent_num]'''

        
        
    for sentence in sentences:
        """First 15 letter is enough for marking a sentence, sentences
        have the same 15 letters are thought to be the same sentence"""
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)

    # 8 Find the average score
    average = _find_average_score(sentence_scores)
    #print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.1*average)
    return summary

def rouge_1(gold_file, file ,Precision=True):
    file = open (file,"r")
    text_str = file.read()
    result = run_summarization(text_str)
    correct_n = 0
    result_tokens = word_tokenize(result)
    file =  open(gold_file,"r")
    real_summary_tokens = word_tokenize(file.read())
    for word in result_tokens:
        if word in real_summary_tokens:
            correct_n += 1
    if Precision:
        acc = correct_n/len(result_tokens)
    else:
        acc = correct_n/len(real_summary_tokens)
    if acc > 1.0:
            print(acc,"====",correct_n,len(result_tokens))
    return acc,result

import matplotlib.pyplot as plt
import numpy as np
import matplotlib           
def draw_plot(data):


    yticksig = range(0,100,5)
    xticksig = np.arange(0,1,0.1)
    plt.yticks(yticksig)
    plt.xticks(xticksig)
    plt.hist(data, bins=40, facecolor="steelblue", edgecolor="black", alpha=1.0)
    
    plt.xlabel("Precision value")
   
    plt.ylabel("Frequecy")
    
    plt.title("Distribution of precision value of tf-idf applied on 500 Business news")
    plt.show()
    
    
    
if __name__ == '__main__':
    '''summaries are store in res_list, check the summary for
    specific summary document by its index
'''
    res_list =[];
    '''evaluated values are store in acc_list, check the value for
    specific summary result by its index
'''
    acc_list =[];
    for i in range(1,501):
        n=str(i)
        file = n.zfill(3)
        
  
        #modify Precision = False to get recall values
        acc,res = rouge_1("business_gold/"+file+".txt","business/"+file+".txt", Precision = True)
        res_list.append(res)
        
        acc_list.append(acc)
        if acc > 1.0:
            print(acc,"====",i)
            
    
        
    print("precision average:",sum(acc_list)/len(acc_list))
    print("e.g.summary of document index 0:\n",res_list[0],"\n")
    print("e.g.precision value of summary index 0: ",acc_list[0],"\n")
    draw_plot(acc_list)   
       
