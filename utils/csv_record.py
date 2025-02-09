import csv

test_fileHeader = ['userid', 'epoch', 'is_poison', 'accuracy', "correct_data", "total_data", 'bengin_mean_acc',
                   'mal_mean_acc']
poison_test_fileHeader = ['userid', 'epoch', 'is_poison', 'accuracy', "correct_data", "total_data", 'bengin_mean_asr',
                          'mal_mean_asr']

test_local_result = []
posiontest_local_result = []
# 记录每一轮本地fine-tune的所有用户的总结果
per_test_result = []
per_posiontest_result = []
# 记录全局模型的结果
global_test_result = []
global_posiontest_result = []

# 记录attack one 的asr
local_poison_test_one_results = []
global_posion_test_one_result = []
per_posiontest_one_result = []


def save_result_csv(folder_path):
    test_csvFile = open(f'{folder_path}/test_local_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_local_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/posiontest_local_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(posiontest_local_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/per_test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(per_test_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/per_posiontest_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(per_posiontest_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/global_test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(global_test_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/global_posiontest_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(global_posiontest_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/local_posiontest_one_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(local_poison_test_one_results)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/global_posiontest_one_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(global_posion_test_one_result)
    test_csvFile.close()

    test_csvFile = open(f'{folder_path}/per_posiontest_one_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(per_posiontest_one_result)
    test_csvFile.close()
