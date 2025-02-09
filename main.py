import copy
import random
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverprox import FedProx
from FLAlgorithms.trainmodel.mnist_model import MnistNet
from FLAlgorithms.trainmodel.fashionmnist_model import FMnistNet
from utils.plot_utils import *
import torch
import datetime

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True  # cudnn
random.seed(1)
np.random.seed(1)


def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, malnum, poisonratio, attack_method,
         per_epoch, attack_start, oneshot, clip_rate, defense):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    trigger_patten = []
    trigger_list = []

    # load trigger
    if dataset == 'Mnist' or dataset == 'FashionMnist':
        for i in range(10, 20, 1):
            for j in range(10, 20, 1):
                trigger_patten.append([i, j])

        malclient = ['f_00007', 'f_00001', 'f_00062', 'f_00020', 'f_00096', 'f_00085', 'f_00051', 'f_00043',
                     'f_00037', 'f_00058']
        poison_label = 1
        intinal_trigger = torch.zeros((1, 28, 28)).float().to(device)

        for i in trigger_patten:
            intinal_trigger[0][i[0]][i[1]] = 0.5

        for i in range(10):
            trigger_list.append(copy.deepcopy(intinal_trigger))

    else:
        raise ValueError("dataset name wrong!")

    print(malclient)

    for i in range(times):  # 重复实验
        print("---------------Running time:------------", i)
        # Generate model

        if dataset == 'Mnist':
            model = MnistNet(name="global", created_time=current_time).to(device)

        elif dataset == 'FashionMnist':
            model = FMnistNet(name="global", created_time=current_time).to(device)

        else:
            raise ValueError("dataset name wrong!")

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                            malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                            attack_method=attack_method, per_epoch=per_epoch, defense=defense)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                             local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                             malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                             attack_method=attack_method, per_epoch=per_epoch, defense=defense)

        else:
            raise ValueError("alg name wrong!")

        final_trigger_list = server.train(pattern=trigger_patten, trigger=trigger_list, per_epoch=per_epoch,
                                          attack_start=attack_start, oneshot=oneshot, clip_rate=clip_rate,
                                          defense=defense)

        print(final_trigger_list[0])

        # local finetuning
        server.send_parameters()  # 将当前的全局模型分给每一个用户 deepcopy

        # Evaluate the final global model
        print("Evaluate the final global model")
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_acc, global_test_mean_mal_acc = server.evaluate()  # 分发了模型，当前用户的模型是全局模型， 输出所有本地数据在全局模型上测试的准确率
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_asr, global_test_mean_mal_asr = server.poison_evaluate(
            trigger=final_trigger_list, pattern=trigger_patten)  # 输出所有本地数据在全局模型上测试的准确率

        print("")

        # Evaluate gloal model on user for each interation
        print("Evaluate the final global model with a few step update, which is personalized model")
        pertestacc, pertrainacc, pertrainloss, poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr, per_mean_mal_asr, per_mean_ben_acc, per_mean_mal_acc = server.evaluate_one_step(
            per_epoch, trigger=final_trigger_list, pattern=trigger_patten)  # 本地finetune一次，在本地模型上再测试   --个性化模型准确率


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10",
                        choices=["Mnist", "FashionMnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn", "VGG16", "resnet", "lenet"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",
                        choices=["pFedMe", "PerAvg-FO", "PerAvg-HF", "FedAvg", "FedProx", "Ditto", "SCAFFOLD", "FedBN"])
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--malclient", type=int, default=10, help="number of malicious client")
    parser.add_argument("--attack_start", type=int, default=10, help="the start attack iteration")
    parser.add_argument("--mal_local_epoch", type=int, default=20)
    parser.add_argument("--poisoning_per_batch", type=int, default=5, help="the poison ratio")
    parser.add_argument("--attack_method", type=str, default='attackall',
                        choices=['attackall'])
    parser.add_argument("--attack_goal", type=str, default='attackall', choices=['attackone', 'attackall'])
    parser.add_argument("--per_epoch", type=int, default='1', help='the epoch for local finetune')
    parser.add_argument("--descrip", type=str, help="the gradient mask ratio")
    parser.add_argument("--oneshot", type=int, default=0, help="one shot attack", choices=[1, 0])
    parser.add_argument("--clip_rate", type=int, default=0, help="one shot attack scale")
    parser.add_argument("--defense", type=str, default='none', help="defense method",
                        choices=['none', 'mkrum', 'trim'])
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Attack method:{}".format(args.attack_method))
    print("Defense method:{}".format(args.defense))
    print("Attack goal:{}".format(args.attack_goal))
    print("Start attack iteration:{}".format(args.attack_start))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Per_local epoch:{}".format(args.per_epoch))
    print("one shot:{}".format(args.oneshot))
    print("scale rate:{}".format(args.clip_rate))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        malnum=args.malclient,
        poisonratio=args.poisoning_per_batch,
        attack_method=args.attack_method,
        per_epoch=args.per_epoch,
        attack_start=args.attack_start,
        oneshot=args.oneshot,
        clip_rate=args.clip_rate,
        defense=args.defense
    )
