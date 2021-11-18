from nltk import Tree
from nltk.tree import ParentedTree
import json

def TextClassDataLoader(path):
    with open(path+'/ReCOVery/test.json') as f1:
        test_data = json.load(f1)
        print("len of test: ", len(test_data))
    with open(path+'/ReCOVery/train.json') as f2:
        train_data = json.load(f2)
        print("len of train: ", len(train_data))
    with open(path+'/ReCOVery/eval.json') as f3:
        val_data = json.load(f3)
        print("len of eval: ", len(val_data))
    print("total_number: ", len(test_data) + len(train_data) + len(val_data))

    test_bodytext = []
    test_root = []
    test_label = []
    test_title = []
    for i in range(len(test_data)):
        news_id = test_data[i]['news_id']
        title = test_data[i]['title']
        reliability = test_data[i]['reliability']

        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r")
        print("--------------------------------------------------------")
        print("processing test news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()

        #getting all cfg
        test_dict={}
        f_read2 = open(path+"/strtree_CGF/news_" + str(news_id+1) + ".txt", "r")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    test_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        test_root.append(news_tree)
        test_bodytext.append(test_dict)
        test_label.append(reliability)
        test_title.append(title)


    train_bodytext = []
    train_root = []
    train_label = []
    train_title = []
    for i in range(len(train_data)):
        news_id = train_data[i]['news_id']
        title = train_data[i]['title']
        reliability = train_data[i]['reliability']

        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r")
        print("--------------------------------------------------------")
        print("processing training news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()


        # getting all cfg
        train_dict = {}
        f_read2 = open(path + "/strtree_CGF/news_" + str(news_id+1) + ".txt", "r")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    train_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        train_root.append(news_tree)
        train_bodytext.append(train_dict)
        train_label.append(reliability)
        train_title.append(title)


    val_bodytext = []
    val_root = []
    val_label = []
    val_title = []
    for i in range(len(val_data)):
        news_id = val_data[i]['news_id']
        title = val_data[i]['title']
        reliability = val_data[i]['reliability']

        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r")
        print("--------------------------------------------------------")
        print("processing val news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()
        # getting all cfg
        val_dict = {}
        f_read2 = open(path + "/strtree_CGF/news_" + str(news_id+1) + ".txt", "r")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    val_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        val_root.append(news_tree)
        val_bodytext.append(val_dict)
        val_label.append(reliability)
        val_title.append(title)

    return test_root,test_bodytext,test_label,test_title,train_root, train_bodytext,train_label,\
           train_title,val_root,val_bodytext,val_label,val_title


if __name__ == '__main__':
    new_data = TextClassDataLoader('./data')
    print("finish")

