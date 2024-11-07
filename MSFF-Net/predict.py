import os
import torch
from PIL import Image
from torchvision import transforms
from models import model
import numpy as np
import sklearn as skl
from sklearn.metrics import accuracy_score
import config
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook



def bootstrap_metrics(metric_func, true_labels, predicted_labels, num_iterations=1000, confidence_level=0.95):
    metrics = []
    for _ in range(num_iterations):
        resampled_true_labels, resampled_predicted_labels = resample(true_labels, predicted_labels)
        metric_value = metric_func(resampled_true_labels, resampled_predicted_labels)
        metrics.append(metric_value)

    mean_metric = np.mean(metrics)
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (confidence_level + (1 - confidence_level) / 2) * 100
    lower_ci = np.percentile(metrics, lower_percentile)
    upper_ci = np.percentile(metrics, upper_percentile)

    return mean_metric, lower_ci, upper_ci


def main():
    mask_path = './mask'
    train_root = './data/train'
    test_root = './data/test_'
    train = []
    test = []

    predict_pro = []
    predict_cla = []
    test_label = []

    label_root = './data/pm_finals.xlsx'
    label_data = pd.read_excel(os.path.join(label_root))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.resnet18().to(device)

    model_weight_path = "Result/weights/ResNet18.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    for root, dirs, files in os.walk(train_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':  # 判断，只记录jpg
                train.append(os.path.join(root, file))

    for root, dirs, files in os.walk(test_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':  # 判断，只记录jpg
                test.append(os.path.join(root, file))


    net.eval()
    with torch.no_grad():
        # predict class
        predict_list= []
        for img_path in test:
            list = []
            img_1 = Image.open(img_path).convert('L')
            i = img_path[13:]
            i = i[:-4]
            img_2 = Image.open(mask_path + '/' + i + '.png').convert('L')
            img_1 = np.array(img_1).astype(float)
            img_2 = np.array(img_2).astype(float)
            img_2[img_2 < 244] = -0.5
            img_2[img_2 > 244] = 0.5
            img_1 = torch.from_numpy(img_1)
            img_2 = torch.from_numpy(img_2)
            resize = transforms.Resize((config.img_w, config.img_h))
            img_1 = torch.unsqueeze(img_1, dim=0)
            img_2 = torch.unsqueeze(img_2, dim=0)
            img_1 = resize(img_1)
            img_2 = resize(img_2)
            img = torch.cat((img_1, img_2), dim=0)
            img = torch.unsqueeze(img, dim=0).float()

            net.fc1.register_forward_hook(get_activation('fc1'))
            output = torch.squeeze(net(img.to(device)))

            label = label_data.loc[label_data['num'] == int(i), ['label']].values.item()

            predict = torch.softmax(output, dim=0)
            # output = torch.max(output, dim=0)[1]
            predict_list.append(predict)
            output = predict[1]

            predict = predict[1]
            threshold = 0.16
            output = torch.where(output >= threshold, torch.ones_like(output), output)
            output = torch.where(output < threshold, torch.zeros_like(output), output)
            predict_class = output.cpu().numpy()
            predict_probability = predict.cpu().numpy()
            predict_pro.append(predict_probability.item())
            predict_cla.append(int(predict_class.item()))
            test_label.append(label)

            list.append(int(i))

            tensor = activation['fc1'].cpu()  # .tolist()
            tensor = torch.squeeze(tensor)
            t = tensor[0].float().item()

            for x in range(0, len(tensor)):
                list.append(tensor[x].float().item())

            # with open("Result/ex_test/2y/ex_2y_L3.csv", "a", newline='', encoding='utf-8') as file:
            #     writer = csv.writer(file, delimiter=',')
            #     writer.writerow(list)
    matrix = skl.metrics.confusion_matrix(test_label, predict_cla, labels=[0, 1])  #
    tn, fp, fn, tp = matrix.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = accuracy_score(test_label, predict_cla)  #
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    AUC = skl.metrics.roc_auc_score(test_label, predict_pro)  #


    # auc, auc_lower_ci, auc_upper_ci = bootstrap_metrics(skl.metrics.roc_auc_score, test_label, predict_pro)
    # acc, acc_lower_ci, acc_upper_ci = bootstrap_metrics(accuracy_score, test_label, predict_cla)
    #
    #
    # # Print out results
    # print('AUC: {:.4f} ({:.4f}, {:.4f})'.format(auc, auc_lower_ci, auc_upper_ci))
    # print('Accuracy: {:.4f} ({:.4f}, {:.4f})'.format(acc, acc_lower_ci, acc_upper_ci))
    # 计算95%置信区间
    y_true = np.array(test_label)
    y_pred = np.array(predict_cla)  # 这里的预测结果需要是类别标签
    y_score = np.array(predict_pro)  # 这里的概率分数是模型的预测概率

    # 使用 bootstrap 方法计算置信区间
    n_bootstraps = 1000  # bootstrap 采样次数
    auc_scores = []
    acc_scores = []
    sen_scores = []
    spe_scores = []
    ppv_scores = []
    npv_scores = []

    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)))
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]
        y_score_sampled = y_score[indices]

        auc_sampled = roc_auc_score(y_true_sampled, y_score_sampled)
        acc_sampled = accuracy_score(y_true_sampled, y_pred_sampled)
        sen_sampled = recall_score(y_true_sampled, y_pred_sampled)
        conf_matrix_sampled = confusion_matrix(y_true_sampled, y_pred_sampled)
        tn_sampled, fp_sampled, fn_sampled, tp_sampled = conf_matrix_sampled.ravel()
        spe_sampled = tn_sampled / (tn_sampled + fp_sampled)
        ppv_sampled = tp_sampled / (tp_sampled + fp_sampled)
        npv_sampled = tn_sampled / (tn_sampled + fn_sampled)

        auc_scores.append(auc_sampled)
        acc_scores.append(acc_sampled)
        sen_scores.append(sen_sampled)
        spe_scores.append(spe_sampled)
        ppv_scores.append(ppv_sampled)
        npv_scores.append(npv_sampled)

    # 计算 95% 置信区间
    alpha = 0.95
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100

    auc_lower_bound = max(0.0, np.percentile(auc_scores, lower_percentile))
    auc_upper_bound = min(1.0, np.percentile(auc_scores, upper_percentile))

    acc_lower_bound = max(0.0, np.percentile(acc_scores, lower_percentile))
    acc_upper_bound = min(1.0, np.percentile(acc_scores, upper_percentile))

    sen_lower_bound = max(0.0, np.percentile(sen_scores, lower_percentile))
    sen_upper_bound = min(1.0, np.percentile(sen_scores, upper_percentile))

    spe_lower_bound = max(0.0, np.percentile(spe_scores, lower_percentile))
    spe_upper_bound = min(1.0, np.percentile(spe_scores, upper_percentile))

    ppv_lower_bound = max(0.0, np.percentile(ppv_scores, lower_percentile))
    ppv_upper_bound = min(1.0, np.percentile(ppv_scores, upper_percentile))

    npv_lower_bound = max(0.0, np.percentile(npv_scores, lower_percentile))
    npv_upper_bound = min(1.0, np.percentile(npv_scores, upper_percentile))

    # print(f"第{index}轮：")
    print('AUC:', AUC)
    print(f"95% 置信区间 (AUC): ({auc_lower_bound:.4f}, {auc_upper_bound:.4f})")
    print('acc: ', acc)
    print(f"95% 置信区间 (Accuracy): ({acc_lower_bound:.4f}, {acc_upper_bound:.4f})")
    print('sensitivity: ', sen)
    print(f"95% 置信区间 (Sensitivity): ({sen_lower_bound:.4f}, {sen_upper_bound:.4f})")
    print('specificity: ', spe)
    print(f"95% 置信区间 (Specificity): ({spe_lower_bound:.4f}, {spe_upper_bound:.4f})")
    print('ppv:', ppv)
    print(f"95% 置信区间 (PPV): ({ppv_lower_bound:.4f}, {ppv_upper_bound:.4f})")
    print('npv:', npv)
    print(f"95% 置信区间 (NPV): ({npv_lower_bound:.4f}, {npv_upper_bound:.4f})")


    # tra_ = '训练集'
    # val_ = '验证集'
    # test_ = '测试集'
    #
    # log_path = './Result/ResNet101.csv'
    # file = open(log_path, 'a+', encoding='utf-8', newline='')
    # csv_writer = csv.writer(file)
    # #csv_writer.writerow([f'数据集', 'AUC', 'ACC', 'SEN', 'SPE', 'PPV', 'NPV'])
    # csv_writer.writerow([test_, "%.3f"%AUC, "%.3f"%acc, "%.3f"%sen, "%.3f"%spe, "%.3f"%ppv, "%.3f"%npv])
    # file.close()
    #
    # fpr,tpr,threshold = roc_curve(test_label, predict_pro)
    # lw = 2
    # plt.figure(figsize=(10,10))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % AUC) ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    # np.save('./roc/fpr-msff-tumor',fpr)
    # np.save('./roc/tpr-msff-tumor',tpr)


if __name__ == '__main__':
    main()
