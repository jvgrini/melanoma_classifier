from Bio import SeqIO
import numpy as np



oneHotNT = {
    'A': [[1,0], [0,0]],
    'T': [[0,1], [0,0]],
    'C': [[0,0], [1,0]],
    'G': [[0,0], [0,1]]
}
oneHot_status = {
    'positive': [1,0],
    'negative': [0,1]
}

positive_data = []
negative_data = []

positive_train_data = []
negative_train_data = []

positive_test_data = []
negative_test_data = []

for seq_record in SeqIO.parse('data/acceptor_positive.fna', 'fasta'):
    positive_data.append([[oneHotNT[letter] for letter in seq_record.seq], [oneHot_status['positive']]])

positive_test_data = positive_data[0:400]
positive_train_data = positive_data[400:]

for seq_record in SeqIO.parse('data/acceptor_negative.fna', 'fasta'):
    negative_data.append([[oneHotNT[letter] for letter in seq_record.seq], [oneHot_status['negative']]])

negative_test_data = negative_data[0:400]
negative_train_data = negative_data[400:]

training_data = np.array(positive_train_data + negative_train_data, dtype=object)
np.random.shuffle(training_data)
np.save('acceptor_training_data.npy', training_data)

test_data = np.array(positive_test_data + negative_test_data, dtype=object)
np.random.shuffle(test_data)
print(len(positive_data))
print(len(positive_train_data))
print(len(test_data))
print(len(training_data))
np.save('acceptor_test_data.npy', test_data)


