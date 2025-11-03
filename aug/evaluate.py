'''
@https://github.com/foresthao/ContrastMatcher 
@mrforesthao
'''
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import time
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from ValGraphMatchDataset.match_dataset import GraphMatchDataset
import matplotlib.pyplot as plt
from sklearn import metrics
import os

@torch.no_grad()
def test_save_model(gconv, dataloader, device, path, epoch,
                    logger, t, norm, save_embed, eval_model, save_model):
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv.state_dict(), path_epo + '_model.pkl')

    gconv.eval()
    x = []
    y = []
    for data in dataloader:
        # print(data.idx)
        data = data.to(device)
        _, g = gconv(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)

    x = torch.cat(x, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None
    
@torch.no_grad()
def test_save_model00(gconv, dataloader, device, path, epoch,
                    logger, t, norm, save_embed, eval_model, save_model, args):  # New code
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv.state_dict(), path_epo + '_model.pkl')

    dataset_test = GraphMatchDataset(args)
    graphs, labels = dataset_test.pack_pair(pack_size=args.batch_size)
    test_dataset = list(zip(graphs, labels))
    random.shuffle(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)


    gconv.eval()
    x1, x2, y =[], [], []
    for data in test_dataloader:
        (graph_t, graph_g), labels = data
        graph_t.to(device)
        graph_g.to(device)
        labels.to(device)
        _, g1 = gconv(graph_t.x, graph_t.edge_index, graph_t.batch)
        _, g2 = gconv(graph_g.x, graph_g.edge_index, graph_g.batch)
        x1.append(g1)
        x2.append(g2)
        y.append(labels)

    x1 = torch.cat(x1, dim=0).cpu().detach().numpy()
    x2 = torch.cat(x2, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()
    x = np.concatenate((x1, x2), axis=1)

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        return test_SVM(x, y, logger, t, norm)
    else:
        return None

@torch.no_grad()
def test_save_model_cm(gconv, dataset, device, path, epoch,
                    logger, t, norm, save_embed, eval_model, save_model, args, plot_on=False, plot_prefix=None, save_roc_data=False):  # New code
    path_epo = path + '_' + str(t) + '_' + str(epoch)

    if save_model:
        torch.save(gconv.state_dict(), path_epo + '_model.pkl')

    dataset_test = GraphMatchDataset(args, dataset=dataset)
    graphs, labels = dataset_test.pack_pair(pack_size=int(len(dataset)))  # Is 0.2
    test_dataset = list(zip(graphs, labels))
    random.shuffle(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)


    gconv.eval()
    x1, x2, y =[], [], []
    for data in test_dataloader:
        (graph_t, graph_g), labels = data
        graph_t.to(device)
        graph_g.to(device)
        labels.to(device)
        _, g1 = gconv(graph_t.x, graph_t.edge_index, graph_t.batch)
        _, g2 = gconv(graph_g.x, graph_g.edge_index, graph_g.batch)
        x1.append(g1)
        x2.append(g2)
        y.append(labels)

    x1 = torch.cat(x1, dim=0).cpu().detach().numpy()
    x2 = torch.cat(x2, dim=0).cpu().detach().numpy()
    y = torch.cat(y, dim=0).cpu().detach().numpy()
    x = np.concatenate((x1, x2), axis=1)

    if save_embed:
        with open(path_epo + '.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f)
        logger.info(path_epo + ' saved.')

    if eval_model:
        # Only use test_SVM2: save to pkl when save_roc_data is True, otherwise only calculate and return metrics
        prefix = (plot_prefix if plot_prefix is not None else path) if save_roc_data else None
        return test_SVM2(x, y, logger, t, norm, save_path_prefix=prefix)
    else:
        return None



def test_LR(x, y, logger, t, norm):
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
    logger.info("Acc:{:.2f}±{:.2f}".format(np.mean(accs), np.std(accs)))
    return np.mean(accs), np.std(accs)


def test_SVM(x, y, logger, t, norm=False, search=True):
    y = LabelEncoder().fit_transform(y)
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    accuracies = []
    accuracies_val = []
    aurocs = []
    aurocs_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        aurocs.append(roc_auc_score(y_test, classifier.decision_function(x_test)) * 100)

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        aurocs_val.append(roc_auc_score(y_test, classifier.decision_function(x_test)) * 100)

    logger.info("accuracies_val:{:.2f}±{:.2f}, accuracies:{:.2f}±{:.2f}".format(
        np.mean(accuracies_val), np.std(accuracies_val), np.mean(accuracies), np.std(accuracies)))
    logger.info("aurocs_val:{:.2f}±{:.2f}, aurocs:{:.2f}±{:.2f}".format(
        np.mean(aurocs_val), np.std(aurocs_val), np.mean(aurocs), np.std(aurocs)))
    return np.mean(accuracies_val), np.mean(accuracies), np.mean(aurocs_val), np.mean(aurocs)


def test_SVM1(x, y, logger, t, norm=False, search=True, save_path_prefix=None):
    '''
    Used to plot ROC curve
    '''
    y = LabelEncoder().fit_transform(y)
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    accuracies = []
    accuracies_val = []
    aurocs = []
    aurocs_val = []
    all_fpr = []  # Store all fpr
    all_tpr = []  # Store all tpr
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        y_score = classifier.decision_function(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)  # Calculate fpr and tpr of ROC curve
        roc_auc = metrics.auc(fpr, tpr)  # Calculate AUC
        aurocs.append(roc_auc_score(y_test, y_score) * 100)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        y_score = classifier.decision_function(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)  # Calculate fpr and tpr of ROC curve
        roc_auc = metrics.auc(fpr, tpr)  # Calculate AUC
        aurocs_val.append(roc_auc_score(y_test, y_score) * 100)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    logger.info("accuracies_val:{:.2f}±{:.2f}, accuracies:{:.2f}±{:.2f}".format(
        np.mean(accuracies_val), np.std(accuracies_val), np.mean(accuracies), np.std(accuracies)))
    logger.info("aurocs_val:{:.2f}±{:.2f}, aurocs:{:.2f}±{:.2f}".format(
        np.mean(aurocs_val), np.std(aurocs_val), np.mean(aurocs), np.std(aurocs)))

    # Plot ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_roc_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % mean_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    outfile = 'roc_curve.pdf' if save_path_prefix is None else f"{save_path_prefix}_roc.pdf"
    plt.savefig(outfile)
    # plt.show()

    return np.mean(accuracies_val), np.mean(accuracies), np.mean(aurocs_val), np.mean(aurocs)


def test_SVM2(x, y, logger, t, norm=False, search=True, save_path_prefix=None):
    '''
    Used to save ROC data without plotting, for subsequent multi-dataset comparison plotting
    '''
    y = LabelEncoder().fit_transform(y)
    if norm:
        minMax = MinMaxScaler()
        x = minMax.fit_transform(x)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=t)
    accuracies = []
    accuracies_val = []
    aurocs = []
    aurocs_val = []
    all_fpr = []  # Store all fpr
    all_tpr = []  # Store all tpr
    all_aucs = []  # Store AUC value for each fold
    
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        y_score = classifier.decision_function(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)  # Calculate fpr and tpr of ROC curve
        roc_auc = metrics.auc(fpr, tpr)  # Calculate AUC
        aurocs.append(roc_auc_score(y_test, y_score) * 100)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_aucs.append(roc_auc)

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)) * 100)
        y_score = classifier.decision_function(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)  # Calculate fpr and tpr of ROC curve
        roc_auc = metrics.auc(fpr, tpr)  # Calculate AUC
        aurocs_val.append(roc_auc_score(y_test, y_score) * 100)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_aucs.append(roc_auc)

    logger.info("accuracies_val:{:.2f}±{:.2f}, accuracies:{:.2f}±{:.2f}".format(
        np.mean(accuracies_val), np.std(accuracies_val), np.mean(accuracies), np.std(accuracies)))
    logger.info("aurocs_val:{:.2f}±{:.2f}, aurocs:{:.2f}±{:.2f}".format(
        np.mean(aurocs_val), np.std(aurocs_val), np.mean(aurocs), np.std(aurocs)))

    # Calculate average ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_roc_auc = metrics.auc(mean_fpr, mean_tpr)
    
    # Save ROC data to file
    if save_path_prefix is not None:
        # Target directory: pkl subdirectory in the same directory as prefix
        prefix_dir = os.path.dirname(save_path_prefix)
        if prefix_dir == '':
            prefix_dir = '.'
        pkl_dir = os.path.join(prefix_dir, 'pkl')
        os.makedirs(pkl_dir, exist_ok=True)
        base_name = os.path.basename(save_path_prefix)
        out_path = os.path.join(pkl_dir, f"{base_name}_roc_data.pkl")

        roc_data = {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'mean_roc_auc': mean_roc_auc,
            'all_fpr': all_fpr,
            'all_tpr': all_tpr,
            'all_aucs': all_aucs,
            'accuracies_val': accuracies_val,
            'accuracies': accuracies,
            'aurocs_val': aurocs_val,
            'aurocs': aurocs
        }
        with open(out_path, 'wb') as f:
            pickle.dump(roc_data, f)
        logger.info(f"ROC data saved to {out_path}")

    return np.mean(accuracies_val), np.mean(accuracies), np.mean(aurocs_val), np.mean(aurocs)
    