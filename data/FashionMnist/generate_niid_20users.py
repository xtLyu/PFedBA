import heapq
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
import json
import os

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

for _, train_data in enumerate(trainloader,0):
    trainset.data, trainset.targets = train_data
for _, train_data in enumerate(testloader,0):
    testset.data, testset.targets = train_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 100 # should be muitiple of 10
NUM_LABELS = 3

# numran1 = random.randint(10, 50)
# numran2 = random.randint(1, 10)
# num_samples = (num_samples) * numran2 + numran1 #+ 100

# Setup directory for train/test data
train_path = './data/train/fashionmnist_train.json'
test_path = './data/test/fashionmnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

MNIST_data_image = []
MNIST_data_label = []

MNIST_data_image.extend(trainset.data.cpu().detach().numpy())
MNIST_data_image.extend(testset.data.cpu().detach().numpy())
MNIST_data_label.extend(trainset.targets.cpu().detach().numpy())
MNIST_data_label.extend(testset.targets.cpu().detach().numpy())
MNIST_data_image = np.array(MNIST_data_image)
MNIST_data_label = np.array(MNIST_data_label)

mnist_data = []
for i in trange(10):
    idx = MNIST_data_label==i
    mnist_data.append(MNIST_data_image[idx])

print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
users_lables = []

print("idx",idx)

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(10):  # 3 labels for each users
        l = (user + j) % 10
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 10

print("IDX1:", idx)  # counting samples for each labels

# # Assign remaining sample by power law
# user = 0
# props = np.random.lognormal(
#     0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
# props = np.array([[[len(v)-NUM_USERS]] for v in mnist_data]) * \
#     props/np.sum(props, (1, 2), keepdims=True)
fenpei=[]

for n in range(10):
    sampled_probabilities = len(mnist_data[n]) * np.random.dirichlet(np.array(NUM_USERS * [0.5]))
    print(sampled_probabilities)
    image_nums = []
    for user in trange(NUM_USERS):
        num_samples = int(round(sampled_probabilities[user]))
        X[user] += mnist_data[n][:min(len(mnist_data[n]), num_samples)].tolist()
        y[user] += (n * np.ones(min(len(mnist_data[n]), num_samples))).tolist()
        image_nums.append(min(len(mnist_data[n]), num_samples))
        mnist_data[n] = mnist_data[n][min(len(mnist_data[n]), num_samples):]
    print(image_nums)
    fenpei.append(image_nums)

for i in fenpei:
    aa=heapq.nlargest(5, range(len(i)), i.__getitem__)
    print(i)
    print(aa)


# for user in trange(NUM_USERS):
#     for j in range(NUM_LABELS):  # 4 labels for each users
#         l = (user + j) % 10
#         num_samples = int(props[l, user//int(NUM_USERS/10), j])
#         numran1 = random.randint(500, 600)
#         num_samples = (num_samples)  + numran1 #+ 200
#         if num_samples > 2000:
#             num_samples = 2000
#         if idx[l] + num_samples < len(mnist_data[l]):
#             X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].values.tolist()
#             y[user] += (l*np.ones(num_samples)).tolist()
#             idx[l] += num_samples
#             print("check len os user:", user, j,
#                   "len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
