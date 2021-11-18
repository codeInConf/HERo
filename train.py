import time
import argparse
from engine import *
import random
from util import *
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
print("torch version :", torch.__version__)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--if_embedding', default=True, type=bool, metavar='N', help='embedding')
parser.add_argument('--cgf_input_dim', default=100, type=int, metavar='N', help='cgf_tree input dim/ word embedding dim')
parser.add_argument('--cgf_output_dim', default=100, type=int, metavar='N', help='cgf_tree output dim/ rst_tree input dim')
parser.add_argument('--cgf_bias', default=False, type=bool, metavar='N', help='For lstm: If False, then the layer does not use bias weights b_ih and b_hh. Default: True')
parser.add_argument('--hidden_dim', default=8, type=int, metavar='N', help='No need for hidden_dim')
parser.add_argument('--rst_input_dim', default=100, type=int, metavar='N', help='rst_tree input dim')
parser.add_argument('--rst_output_dim', default=100, type=int, metavar='N', help='rst_tree output dim')
parser.add_argument('--n_layers', default=1, type=int, metavar='N', help='number of output classes')
parser.add_argument('--rst_bias', default=False, type=bool, metavar='N', help='For lstm: If False, then the layer does not use bias weights b_ih and b_hh. Default: True')
parser.add_argument('--rst_drop_prob', default=0.2, type=float, metavar='N', help='dropout for rst')
parser.add_argument('--rst_bidirect', default=True, type=bool, metavar='N', help='number of output classes')
parser.add_argument('--cgf_drop_prob', default=0.2, type=float, metavar='N', help='dropout for cgf')
parser.add_argument('--cgf_bidirect', default=True, type=bool, metavar='N', help='number of output classes')
parser.add_argument('--classes', default=2, type=int, metavar='N', help='number of output classes')
parser.add_argument('--max_depth', default=100, type=int, metavar='N', help='max depth for the tree')
parser.add_argument('--max_child', default=20, type=int, metavar='N', help='max child # for the node in cgf')
parser.add_argument('--clip', default=2, type=int, metavar='N', help='')
parser.add_argument('--print_every',type=int,default=10,help='')
parser.add_argument('--save', type = str, default="result/", help='')

args = parser.parse_args()


def pre_label_helper(pre, label):
    b = 0 if label[0] == 1 else 1
    a = 0 if pre[0] > pre[1] else 1

    return a, b

def main():

    device_setting = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device_setting)
    print("===> loading  embed ...")

    embed = {}
    f = open('glove.6B.100d.txt', encoding="utf8")
    for i in f:
        value = i.split()
        word = value[0]
        embed[word] = np.asarray( value[1:], dtype='float32')
    f.close()
    embeding_example = embed['the']
    print("embeding_example: ", embeding_example)
    print(type(embeding_example))

    # create trainer
    print("===> creating dataloaders ...")
    end = time.time()
    data_loader = TextClassDataLoader('./data')
    print('===> dataloader creatin: {t:.3f}'.format(t=time.time() - end))
    test_root = data_loader[0]
    test_bodytext = data_loader[1]
    test_label = data_loader[2]
    test_title = data_loader[3]

    train_root = data_loader[4]
    train_bodytext = data_loader[5]
    train_label = data_loader[6]
    train_title = data_loader[7]

    val_root = data_loader[8]
    val_bodytext = data_loader[9]
    val_label = data_loader[10]
    val_title = data_loader[11]

    from collections import Counter
    print(Counter(train_label))
    print(Counter(val_label))
    print(Counter(test_label))


    print("train size: " , len(train_root))
    print("val size: " , len(val_root))
    print("test size: " , len(test_root))


    # create model
    print("===> creating rnn model ...")
    engine = trainer(max_depth=args.max_depth, max_number_child=args.max_child, device=device_setting, embed_dict=embed,
                    cgf_input_dim=args.cgf_input_dim, cgf_output_dim=args.cgf_output_dim, cgf_bias=args.cgf_bias, hidden_dim=args.hidden_dim,
                    rst_input_dim=args.rst_input_dim, rst_output_dim=args.rst_output_dim, n_layers=args.n_layers, rst_bidirect=args.rst_bidirect,
                    cgf_drop_prob=args.cgf_drop_prob, cgf_bidirect=args.cgf_bidirect, clip=args.clip, lrate=args.lr, wdecay=args.weight_decay)

    #training
    f = open("result/result_childEndCut_TreeAverage.txt", "w")
    print("args:", args)
    log = 'args: lr: {:05f}, max_depth: {}, max_child: {}\n'
    print(log.format(args.lr,args.max_depth, args.max_child, flush=True))

    f.write(log.format(args.lr,args.max_depth, args.max_child))
    print("start training...", flush=True)
    his_loss = []
    his_auc = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):


        train_loss = []
        train_acc=[]
        t_pre=[]
        t_label=[]
        t1 = time.time()
        train_idx = np.arange(0, len(train_root)).tolist()
        random.shuffle(train_idx)
        for iter, idx in enumerate(train_idx):

            tr_root = train_root[idx]
            tr_bodytext = train_bodytext[idx]
            if train_label[idx] == 0:

                tr_label = torch.Tensor([1,0]).to(device_setting)
            else:
                tr_label = torch.Tensor([0, 1]).to(device_setting)

            metrics, (pre, label) = engine.train(tr_root, tr_bodytext, tr_label)

            train_loss.append(metrics[0])

            pre, label = pre_label_helper(pre, label)

            t_label.append(label)

            t_pre.append(pre)

            train_acc.append(metrics[1])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train acc: {:.4f}'
                print(log.format(iter, train_loss[-1], train_acc[-1], flush=True))
        t2 = time.time()
        train_time.append(t2 - t1)
        log = 'Epoch: {:03d}, Train Time: {:.4f} secs'
        print(log.format(i, (t2 - t1)))

        # validation
        v_pre = []
        v_label = []
        valid_loss = []
        valid_acc = []
        val_idx = np.arange(0, len(val_root)).tolist()
        random.shuffle(val_idx)
        s1 = time.time()
        for iter, idx in enumerate(val_idx):

            vl_root = val_root[idx]
            vl_bodytext = val_bodytext[idx]
            if val_label[idx] == 0:

                vl_label = torch.Tensor([1, 0]).to(device_setting)
            else:
                vl_label = torch.Tensor([0, 1]).to(device_setting)

            metrics, (pre, label) = engine.eval(vl_root, vl_bodytext, vl_label)
            valid_loss.append(metrics[0])
            valid_acc.append(metrics[1])

            pre, label = pre_label_helper(pre, label)

            v_label.append(label)

            v_pre.append(pre)

        s2 = time.time()
        log = 'Epoch: {:03d}, Val Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_acc = np.mean(train_acc)

        mvalid_loss = np.mean(valid_loss)
        mvalid_acc = np.mean(valid_acc)
        his_loss.append(mvalid_loss)



        f.write("t_label:"+str(Counter(t_label)))
        f.write("t_pre:" + str(Counter(t_pre)))
        f.write("v_pre:" + str(Counter(v_pre)))
        f.write("v_label:" + str(Counter(v_label)))

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train ACC: {:.4f}, Train roc_auc_score_weighted: {:.4f}, Valid Loss: {:.4f}, Valid ACC: {:.4f}, Val roc_auc_score_weighted: {:.4f}, Training Time: {:.4f}/epoch\n'
        print(log.format(i, mtrain_loss, mtrain_acc, roc_auc_score(t_label, t_pre, average="weighted"), mvalid_loss, mvalid_acc, roc_auc_score(v_label, v_pre, average="weighted"), (t2 - t1)),
              flush=True)
        val_auc = roc_auc_score(v_label, v_pre, average="weighted")
        his_auc.append(val_auc)
        f.write(log.format(i, mtrain_loss, mtrain_acc, roc_auc_score(t_label, t_pre, average="weighted"), mvalid_loss, mvalid_acc, roc_auc_score(v_label, v_pre, average="weighted"), (t2 - t1)))
        print()
        torch.save(engine.model.state_dict(), args.save + "valid_loss_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
        torch.save(engine.model.state_dict(), args.save + "valid_auc_epoch_" + str(i) + "_" + str(round(val_auc, 2)) + ".pth")
    f.write("Average Training Time: {:.4f} secs/epoch\n".format(np.mean(train_time)))
    f.write("Average Inference Time: {:.4f} secs\n".format(np.mean(val_time)))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    # testing max val auc
    bestid = np.argmax(his_auc)
    engine.model.load_state_dict(torch.load(args.save + "valid_auc_epoch_" + str(bestid + 1) + "_" + str(round(his_auc[bestid], 2)) + ".pth"))

    ttt_pre = []
    ttt_label = []
    test_time = []
    test_idx = np.arange(0, len(test_root)).tolist()
    random.shuffle(test_idx)
    s1 = time.time()
    for iter, idx in enumerate(test_idx):
        tt_root = test_root[idx]
        tt_bodytext = test_bodytext[idx]
        if test_label[idx] == 0:
            tt_label = torch.Tensor([1, 0]).to(device_setting)
        else:
            tt_label = torch.Tensor([0, 1]).to(device_setting)
        with torch.no_grad():
            preds = engine.model(tt_root, tt_bodytext)[0][0]

        preds = preds.tolist()
        label = tt_label.tolist()
        pre, label = pre_label_helper(preds, label)

        ttt_label.append(label)

        ttt_pre.append(pre)

    f.write("ttt_label:" + str(Counter(ttt_label)))
    f.write("ttt_pre: " + str(Counter(ttt_pre)))

    s2 = time.time()
    log = 'From Max AUC, Test roc_auc_score_weighted: {:.4f}, Test Time: {:.4f}\n'
    print(log.format(roc_auc_score(ttt_label, ttt_pre, average="weighted"), (s2 - s1)), flush
          =True)
    f.write(log.format(roc_auc_score(ttt_label, ttt_pre, average="weighted"), (s2 - s1)))
    f.close()


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))




