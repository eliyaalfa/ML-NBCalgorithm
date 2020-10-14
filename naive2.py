import csv
import math
import random
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from matplotlib.pylab import plt
from sklearn.metrics import confusion_matrix

myProbabilities = dict()

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
            #dataset isinyaa dalah dataset .csv
            #print("dataset : %s" % row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        # row isinyaa dalah dataset .csv
        # print("row : %s" % row)
        row[column] = float(row[column].strip())
        #print(row[column])

#convert string column to integer
def str_column_to_int(dataset,column):
    #variabel class_value menyimpan data kolom aktivitas (berhenti,lurus,dll)
    class_values = [row[column] for row in dataset]
    print("class_values %s" % class_values)
    unique = set(class_values)  # salah satu tipe data di python yang tidak berurut. Set memiliki anggota yang unik (tdk ada duplikasi)
    lookup = dict()  # berfungsi untuk membuat dictionary, apabila kosong akan mengembalikan None
    for i, value in enumerate(unique):  # mengembalikan obyek iterable yang tiap itemnya berpasangan dengan indeks
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
        #row[column] berisi nilai label aktivitas dari 0-3
        print("row[column] % s" % row[column])
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()#mengembalikan list berisi anggota-anggota dari obyek yang menjadi argumennya. Jika argumen kosong maka mengembalikan list kosong
    dataset_copy = list(dataset)
    print(dataset_split)
    print(dataset_copy)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        #print("i : %s" % i)
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            #print(index)
            #index membangkitkan nilai random
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
        #print("dataset_split : %s" % dataset_split)
        #datasplit berisi array data yang telah dibagi dalam kfold
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        #print("i : %s" % i)
        if actual[i] == predicted[i]:
            correct += 1
            #coorect akan berisi nilai 1 hingga = 4 sebanyak n(3) + 1
            #print("correct : %s" % correct)
    return correct / float(len(actual)) * 100.0

#evaluate algoritma
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        #variabel train_set menyimpan nilai dari datasplit diatas
        for tr in train_set:
            print("Train Set : %s" % train_set)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        #variabel row merupaan isi dari variabel fold yang berisi kumpulan dataset yang telah dibagi
        print("roww")
        print(row)
        print("fold")
        print(fold)
        accuracy = accuracy_metric(actual, predicted)
        results = confusion_matrix(actual, predicted)
        scores.append(accuracy)
        for a in test_set:
            print("Test set: %s" %a)
        print("Actual    : %s" % actual)
        print("predicted : %s" % predicted)
        print("Result : " )
        print(results)
        plt.bar(n_folds, accuracy)
    return scores

#split dataset by class values
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        #variabel vector berisi nilai dataset yang diacak (misal sejumlah 16 dari 20)
        #print("vector %s" % vector)
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    print ("Separated_Class %s" % separated)
    return separated


#hitung mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))

#hitung standard deviasi
def stdev(numbers):
    avg = mean(numbers)
    #numbers membawa nilai tiap fitur tiap kondisi yang ditampilkan dalam bentuk kolom[.....]
    print(numbers)
    variance = sum([math.pow(x - avg, 2) for x in numbers])/float(len(numbers)-1)
    for x in numbers:
        #x merupakan nilai yang akan dihitung nilai mean dan varian, terdiri dari nilai tiap feature dalam kondisi yang samaa
        #x adalah member dari variable numbers, ditampilkan ke bawah dalam bentuk baris
        print(x)
    print("mean   : %s" % avg)
    print("varian : %s" % variance)
    return math.sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
#fungsi zip disini untuk mendapatkan nilai stdev dan mean each kolom, supaya jadii satu
def summarize_dataset(dataset):
    print ("myDataset : %s" % dataset)
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    for column in zip(*dataset):
        #variabel colomn menyimpan data nilai tiap fiturr x,y,z dari baris dijadikan kolom kesamping
        print(column)
    del summaries[-1]
    #menghapus nilai summaries satu persatu
    #print(summaries)
    return summaries

#split data by class itung statistik tiap row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    print("My_Summerize = %s" %summaries)
    return summaries

#hitung Gaussian PDF
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    print(x)
    print(mean)
    print(stdev)
    print("exponent = %s" % exponent)
    pdf = (1/(math.sqrt(2*math.pi)*stdev))*exponent
    print("pdf = %s" % pdf)
    return pdf

#hitung probabilitas kelas
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

#make prediction
def predict(summaries, row):
    myProbabilities = calculate_class_probabilities(summaries, row)
    print("probabilities : %s" %myProbabilities)
    best_label, best_prob = None, -1
    for class_value, probability in myProbabilities.items():
        print("probability %s" % probability)
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
            print("Best Label : %s" % best_label)
    return best_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    #summarize itu sama kek codebook
    summarize = summarize_by_class(train)
    total_rows = sum([summarize[label][0][2] for label in summarize])
    #evidence
    # prob = calculate_class_probabilities(summaries, row) / float(total_likelihood)
    # print("probabilitas : %s" %prob)

    print ("Total Row : %s" % total_rows)
    predictions = list()
    print("prediction: %s" % predictions)
    for row in test:
        #print("test: %s" %test)
        output = predict(summarize, row)
        predictions.append(output)
        #SUMMARIZE BERISI NILAI (MEAN, STDEV, panjang kelas(banyak membernya))
        print("Codebook NBC : %s" % summarize)
        print("Summarize : %s" %summarize)
        myProbabilities = calculate_class_probabilities(summarize, row)
        sum([myProbabilities[label] for label in myProbabilities])
    return (predictions)


seed(1)
#load and prepare data
filename = "E:\GAMBARTREND\ALPHA01-09\SAMSUNG\samsung.csv"
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0]) - 1)
n_folds = 9
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))