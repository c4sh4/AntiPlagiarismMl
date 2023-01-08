"""Creating a dataframe using functions from compare.py.
    Dataframe markup. Training the RandomForestClassifier model.
    Conservation of the model and creation of a pkl file.
"""
import pickle
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import compare as c
# from sklearn.metrics import accuracy_score


def parser_train():
    """ Allows you to run train from a terminal."""
    pars = argparse.ArgumentParser(description='')
    pars.add_argument('files', type=str, help='Input file with a list of file pairs to check.')
    pars.add_argument('plagiat1', type=str, help='Input file with a list of file pairs to check.')
    pars.add_argument('plagiat2', type=str, help='Input file with a list of file pairs to check.')
    args = pars.parse_args()
    source_t = (args.files, args.plagiat1, args.plagiat2)
    return source_t


def create_df(res_list, mark):
    """ Lazy creation of training dataframe."""
    b_dl = []
    b_lcs = []
    t_dl = []
    t_lcs = []
    l_dl = []
    l_lcs = []
    m_list = []
    for res_ in enumerate(res_list):
        b_dl.append(res_list[res_[0]][0][0])
        b_lcs.append(res_list[res_[0]][0][1])
        t_dl.append(res_list[res_[0]][1][0])
        t_lcs.append(res_list[res_[0]][1][1])
        l_dl.append(res_list[res_[0]][2][0])
        l_lcs.append(res_list[res_[0]][2][1])
        m_list.append(mark)
    pr_dict = {'b_dl': b_dl, 'b_lcs': b_lcs, 't_dl': t_dl, 't_lcs': t_lcs,
               'l_dl': l_dl, 'l_lcs': l_lcs, 'mark': m_list}
    marked_df = pd.DataFrame(pr_dict)
    return marked_df


def mark_for_pairs(file_path, plagiat1_path, plagiat2_path):
    """ Formation and marking of a dataframe for model training."""
    f_l = []  # list of paths in file
    p1_l = []  # list of paths in plagiat1
    p2_l = []  # list of paths in plagiat2
    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as file_:
            f_l.append(str(file_path + "\\" + str(filename)))

    for plagiat1 in os.listdir(plagiat1_path):
        with open(os.path.join(plagiat1_path, plagiat1), 'r', encoding='utf-8') as p_1:
            p1_l.append(str(plagiat1_path + "\\" + str(plagiat1)))

    for plagiat2 in os.listdir(plagiat2_path):
        with open(os.path.join(plagiat2_path, plagiat2), 'r', encoding='utf-8') as p_2:
            p2_l.append(str(plagiat2_path + "\\" + str(plagiat2)))
    pair_train_n = []  # lists of pair true - negative (n - negative) ->1
    pair_train_p = []  # lists of pair true - true (p - positive) -> 0
    for p in enumerate(f_l):
        pair_train_n.extend([f_l[p[0]], p2_l[p[0]], p1_l[p[0]], p2_l[p[0]]])
        pair_train_p.extend([f_l[p[0]], f_l[-p[0] - 1], p1_l[p[0]], p1_l[-p[0] - 1]])
    # print(pair_train_n, pair_train_p, sep='\n')
    pair_file_n = []  # true - negative files
    pair_file_p = []  # true - true files
    for j1, pth1 in enumerate(pair_train_n[:-1]):
        if j1 % 2 == 0:
            pair_n = [c.get_file(pth1), c.get_file(pair_train_n[j1 + 1])]
            pair_file_n.append(tuple(pair_n))
    for j2, pth2 in enumerate(pair_train_p[:-1]):
        if j2 % 2 == 0:
            pair_p = [c.get_file(pth2), c.get_file(pair_train_p[j2 + 1])]
            pair_file_p.append(tuple(pair_p))
    score_list_n = []
    for iter_n, pair_n in enumerate(pair_file_n):
        score_list_n.append(c.get_scores(pair_n))
    # create a dataframe
    final_df = create_df(score_list_n, 1)
    score_list_p = []
    for iter_p, pair_p in enumerate(pair_file_p):
        score_list_p.append(c.get_scores(pair_p))
    final_df = pd.concat([final_df, create_df(score_list_p, 0)], ignore_index=True)
    # df.to_pickle("scoring_mark.pkl")
    return final_df


def model_training(df):
    """ Random forest training"""
    x_col = ['b_dl', 'b_lcs', 't_dl', 't_lcs', 'l_dl', 'l_lcs']
    y_col = ['mark']
    x = df[x_col]
    y = df[y_col]
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values,
                                                        train_size=0.8,
                                                        random_state=42)
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(max_depth=2, random_state=0)
    )
    clf = clf.fit(x_train, y_train.ravel())
    # pred_train = clf.predict(x_train)
    # print('accuracy is: ', accuracy_score(pred_train, y_train))
    # print('predict for test:', clf.predict([[1.000, 1.000, 0.699, 0.699, 0.379, 0.574]]))
    return clf


if __name__ == "__main__":
    p_source = parser_train()
    print(type(p_source[2]), p_source[2], sep='\n')
    df_ = mark_for_pairs(p_source[0], p_source[1], p_source[2])
    clf_ = model_training(df_)
    pickle.dump(clf_, open('model.pkl', 'wb'))
