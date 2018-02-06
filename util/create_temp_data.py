import csv

import tqdm


def mnist_temp():
    with open('../dataset/rating.csv', 'r') as f:
        reader = csv.reader(f)
        print('before np.array')

        temp = []
        print('working on temp')
        for index, rating in tqdm.tqdm(enumerate(list(reader))):
            if index < 100000:
                temp.append(rating)

        print('saving to file')
        with open('../data/rating.csv', 'w') as the_file:
            for line in tqdm.tqdm(temp):
                the_file.write(line[0]+','+line[1]+','+line[2] + '\n')


def small_temp():
    with open('../dataset/small/ratings.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        print('before np.array')

        temp = []
        print('working on temp')
        for index, rating in tqdm.tqdm(enumerate(list(reader))):
            temp.append(rating)

        print('saving to file')
        with open('../data/small_rating.csv', 'w') as the_file:
            for line in tqdm.tqdm(temp):
                the_file.write(line[0]+','+line[1]+','+line[2] + '\n')


small_temp()
