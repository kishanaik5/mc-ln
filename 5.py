import csv
def find_s(training_data):
    hypo = []
    hypo = training_data[0][-1]
    for ex in training_data:
        features = ex[:-1]
        label = ex[-1]
        if label == 'yes':
            for i in range (len(hypo)):
                if hypo[i]!= features[i]:
                    hypo[i] = '?'
            print(hypo)
    return hypo
training_data = []
with open('enjoysport.csv','r') as file:
    csv_read = csv.reader(file)
    for row in csv_read:
        training_data.append(row)
    print(training_data)
    training_data.pop(0)
    print(training_data)
        
h = find_s(training_data)
print("most specific hypothesis is:",h)
