import pandas as pd
from random import seed
from random import randrange
from csv import*
from math import sqrt
from matplotlib.pylab import plt
from sklearn.metrics import confusion_matrix
from decimal import*

tempIndex = []
unique = []

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    global unique
    unique = set(class_values)  #salah satu tipe data di python yang tidak berurut. Set memiliki anggota yang unik (tdk ada duplikasi)
    #mengidentifikasi label kelas pada kolom terakhir data
    print("unigue : %s" % unique)
    #print("Kelas : %s" % class_values)
    lookup = dict() #berfungsi untuk membuat dictionary, apabila kosong akan mengembalikan None
    print("lookup : %s" % lookup)
    for i, value in enumerate(unique): #mengembalikan obyek iterable yang tiap itemnya berpasangan dengan indeks
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list() #mengembalikan list berisi anggota-anggota dari obyek yang menjadi argumennya. Jika argumen kosong maka mengembalikan list kosong
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        #print("Train Set : %s" % train_set)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        results = confusion_matrix(actual, predicted)
        scores.append(accuracy)
        print("Test set: %s" % test_set)
        #print("Actual       : %s" % actual)
        #print("accuracy : %s" %accuracy)
        print("Result : %s" %results)
        #plt.bar(n_folds, accuracy)
        #plt.show()
    return scores


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = Decimal(0)
    for i in range(len(row1) - 1):
        distance += (Decimal(row1[i]) - Decimal(row2[i])) ** Decimal(2)
    return Decimal(sqrt(distance))

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        #print("Test_row : %s" % test_row)
        #print("Distance : %s" % dist)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: Decimal(tup[1]))
    return distances[0][0]
'''
def write_codebooks(codebooks):
    with open('codebook_awal_training_LVQ_sensor_all_acc_smoothing01_code200.csv', 'a+', newline='\n') as csvfile:
        spamwriter = writer(csvfile, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        spamwriter.writerow(codebooks)
'''
# Make a prediction with codebook vectors
def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    #print("Codebook : %s" %codebooks)
    #print("BMU  : %s" %bmu)
    return bmu[-1]

#Mengambil parameter codebooks untuk setiap kelas data
def random_Index_Train(n):
    myRand = randrange(n)
    if len(tempIndex) != n:
        for index in tempIndex:
            if tempIndex[index] == myRand:
                random_Index_Train(n)
            else:
                tempIndex.append(myRand)
    return myRand

# Create a random codebook vector
def random_codebook(train):
    n_records = len(train)
    #print("Jumlah Records : %s" %n_records)
    n_features = len(train[0])
    #print("Jumlah Features : %s" % n_features)
    codebook = train[random_Index_Train(n_records)]
    #write_codebooks(codebook)
    return codebook


def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    print("Codebooks train : %s" %codebooks)
    for epoch in range(epochs):
        #print("Epoch : %s" %epoch)
        #print("Epochs : %s" %epochs)
        rate = lrate * (1.0 - (epoch / float(epochs)))
        #print("rate : %s" %rate)
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            #print("Codebooks : %s" %codebooks)
            #print("Row : %r" %row)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
                    #print("BMU1 : %s" %bmu)

    return codebooks


# LVQ Algorithm
'''
def write_hasil_codebook(codebooks):
    with open('codebook_hasil_training_LVQ_sensor_all_acc_smoothing01_code200.csv', 'w', newline='\n') as csvfile:
        spamwriter = writer(csvfile, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        for item in codebooks:
            spamwriter.writerow(item)
'''

def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    #write_hasil_codebook(codebooks)
    predictions = list()
    print("Codebooks: %s" % codebooks)
    print("Unique %s" % unique)
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
        print("Output : %s" % output)
        for i, value in enumerate(unique):
            if i == output:
                print("Output : %s" % value)
    return (predictions)


# Test LVQ on Iris dataset
seed(1)
# load and prepare data
filename = 'sensor.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 9
learn_rate = 0.1
n_epochs = 100
n_codebooks = 200
scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))