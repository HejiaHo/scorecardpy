# -*- coding: utf8 -*-
# Traditional Credit Scoring Using Logistic Regression
from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd

import scorecardpy as sc

pd.set_option('display.max_columns', 500)
save_path = 'C:/Users/hjse7/Documents/github/scorecardpy/'

if __name__ == '__main__':
    # data prepare ------
    # load germancredit data
    dat = sc.germancredit()
    # filter variable via missing rate, iv, identical value rate
    dt_s = sc.var_filter(dat, y="creditability")

    # breaking dt into train and test
    train, test = sc.split_df(dt_s, 'creditability').values()

    # woe binning ------
    bins = sc.woebin(dt_s, y="creditability")
    # print bins

    plotlist = sc.woebin_plot(bins)

    # # save binning plot
    # for key,i in plotlist.items():
        # i.show()
        # i.savefig(save_path + str(key)+'.png')

    # binning adjustment
    # # adjust breaks interactively
    # breaks_adj = sc.woebin_adj(dt_s, "creditability", bins)
    # # or specify breaks manually
    breaks_adj = {
        'age.in.years': [26, 35, 40],
        'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
    }
    bins_adj = sc.woebin(dt_s, y="creditability", breaks_list=breaks_adj)
    # print bins_adj

    # converting train and test into woe values
    train_woe = sc.woebin_ply(train, bins_adj)
    test_woe = sc.woebin_ply(test, bins_adj)
    # print (train_woe)


    y_train = train_woe.loc[:,'creditability']
    X_train = train_woe.loc[:,train_woe.columns != 'creditability']
    y_test = test_woe.loc[:,'creditability']
    X_test = test_woe.loc[:,train_woe.columns != 'creditability']

    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
    # print (X_train)
    lr.fit(X_train, y_train)
    # print (lr.coef_)
    # print (lr.intercept_)

    # predicted proability
    train_pred = lr.predict_proba(X_train)[:,1]
    test_pred = lr.predict_proba(X_test)[:,1]
    # performance ks & roc ------
    train_perf = sc.perf_eva(y_train, train_pred, title = "train")
    test_perf = sc.perf_eva(y_test, test_pred, title = "test")

    # score ------
    card = sc.scorecard(bins_adj, lr, X_train.columns)
    for key, value in card.iteritems():
        print(key)
        print('-' * 20)
        print(value)
        print()
    # print(card)
    # credit score
    train_score = sc.scorecard_ply(train, card, print_step=0)
    test_score = sc.scorecard_ply(test, card, print_step=0)

    # psi
    sc.perf_psi(
      score = {'train':train_score, 'test':test_score},
      label = {'train':y_train, 'test':y_test}
    )