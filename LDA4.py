import string

import numpy as np

from nltk.corpus import stopwords


def loader(file_name):
    file = open(file_name, "r")
    r = file.read()
    r = r.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
    file_token = r.split()
    file_token_lower = list(map(str.lower, file_token))  # converting the document into lowercase
    file_token_lower_stop_words = [word for word in file_token_lower if
                                   not word in stopwords.words()]  # removing stop words
    return file_token_lower_stop_words


doc_1 = loader("doc_1.txt")
doc_2 = loader("doc_2.txt")
doc_3 = loader("doc_3.txt")
doc_4 = loader("doc_4.txt")
tokenized_docs = [doc_1, doc_2, doc_3, doc_4]

print("tokenized_docs: ", tokenized_docs)
# define the vocabulary
vocab = list(set([word for doc in tokenized_docs for word in doc]))
# print("vocab: ",vocab)
# define the number of topics and iterations
num_topics = 4
num_iterations = 100

topic_word = np.zeros((len(vocab), num_topics))
doc_topic = np.zeros((len(tokenized_docs), num_topics))

word_topic = []
for i, doc in enumerate(tokenized_docs):
    doc_topic_dist = np.random.dirichlet(np.ones(num_topics))
    for j, word in enumerate(doc):
        topic = np.random.choice(num_topics, p=doc_topic_dist)
        word_topic.append((i, j, topic))
        topic_word[vocab.index(word), topic] += 1
        doc_topic[i, topic] += 1

alpha = 0.01
beta = 0.1

for iteration in range(num_iterations):
    for i, j, topic in word_topic:
        word = tokenized_docs[i][j]
        topic_word[vocab.index(word), topic] -= 1
        doc_topic[i, topic] -= 1
        topic_dist = (topic_word[vocab.index(word), :] + beta) * (doc_topic[i, :] + alpha) / (np.sum(topic_word, axis=0) + beta) / (np.sum(doc_topic[i, :]) + alpha)
        normalized_topic_dist = topic_dist / np.sum(topic_dist)
        new_topic = np.random.choice(num_topics, p=normalized_topic_dist)
        t = (i * len(tokenized_docs[i])) + j
        index = 0
        if i > 0:
            for p in range(i):
                index += len(tokenized_docs[p])
        else:
            index = 0

        # word_topic[(i * len(tokenized_docs[i])) + j] = (i, j, new_topic)
        word_topic[(index) + j] = (i, j, new_topic)
        topic_word[vocab.index(word), new_topic] += 1
        doc_topic[i, new_topic] += 1

# print the top words for each topic
print("The top words in the topics are:")
for topic in range(num_topics):
    top_words = [vocab[i] for i in np.argsort(topic_word[:, topic])[::-1][:10]]
    print('Topic {}: {}'.format(topic + 1, ' '.join(top_words)))
print()

for i in range(len(doc_topic)):
    normalised_doc_topic = doc_topic[i] / np.sum(doc_topic[i]) * 100
    print("The percentages of topics in doc no. " + str(i + 1) + "is: ")
    for j in range(len(doc_topic[i])):
        print("Topic " + str(j + 1) + ": " + str(round(normalised_doc_topic[j], 1)) + " %")
    print()
