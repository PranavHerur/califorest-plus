import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
import math
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pylab as plt

"""
对超参数进行二进制编码得到染色体：
n_estimators 取值范围 {10、20、30、40、50、60、70、80、90、100、110、120、130、140、150、160}
max_depth 取值范围 {1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16} 
max_features 取值范围 {3，4，5，6}
（****，****，**）基因组10位长
"""

# 设置遗传代数：
generations = 10
# 种群数量：
populations = 40
# 染色体长度
chrom_len = 10
# 交叉概率
copulation = 0.4
# 变异概率
variation = 0.01
# 存储每一代的平均得分
avg_scores = []
# 存储每一代的最高得分
max_scores = []
# 个体适应度
fit_value = []


# 根据参数生成随机森林，并且返回精度得分
def get_RandomForest(
    n_estimators_value, max_depth_value, max_features_value, train, test
):
    train_y = train["VCI"]
    train_x = train.values[:, :-1]
    test_x = test.values[:, :-1]
    test_y = test["VCI"]
    rf = RandomForestClassifier(
        n_estimators=n_estimators_value,
        max_depth=max_depth_value,
        max_features=max_features_value,
        n_jobs=-1,
    )
    rf.fit(train_x, train_y)
    return rf.score(test_x, test_y)


# 初始化每一个个体的基因，随机化编码
def code_chrom(pop_size, chrom_len):
    pop = []
    for i in range(pop_size):
        tmp = []
        for j in range(chrom_len):
            tmp.append(random.randint(0, 1))
        pop.append(tmp)
    return pop


# 对每个个体进行解码，返回 n_estimators 和 max_depth 和 max_features
def decode_chrom(pop):
    parameters = []
    for i in range(len(pop)):
        res = []
        # 计算n_estimators:
        a = pop[i][0:4]
        first = 0
        for j in range(4):
            first += a[j] * (math.pow(2, j))
        res.append(int(first))
        # 计算max_depth:
        b = pop[i][4:8]
        second = 0
        for j in range(4):
            second += b[j] * (math.pow(2, j))
        res.append(int(second))
        # 计算max_features:
        c = pop[i][8:10]
        third = 0
        for j in range(2):
            third += c[j] * (math.pow(2, j))
        res.append(int(third))
        parameters.append(res)
    return parameters


# 计算每个个体的目标函数值
def com_objvalue(pop, train, test):
    objvalue = []
    values = decode_chrom(pop)
    for i in range(len(values)):
        tmp = values[i]
        n_estimators_value = (tmp[0] + 1) * 10
        max_depth_value = tmp[1] + 1
        max_features_value = tmp[2] + 1 + 3
        score = get_RandomForest(
            n_estimators_value, max_depth_value, max_features_value, train, test
        )
        objvalue.append(score)
    return objvalue


# 计算每个个体的适应度值
def com_fitvalue(obj_value):
    fit_value = []
    tmp = 0.0
    b = 0
    for i in range(len(obj_value)):
        if obj_value[i] + b > 0:
            # 由于原目标函数值相差较小，所以映射到指数函数，增大差距
            tmp = b + obj_value[i]
            tmp = tmp * 10
            tmp = math.exp(tmp)
        else:
            tmp = 0.0
        fit_value.append(tmp)
    return fit_value


# 找出目标函数中最大的和相应的个体
def get_best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if fit_value[i] > best_fit:
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# 记录最好的结果的参数
def record_best(best_individual):
    tmp = []
    tmp.append(best_individual)
    value = decode_chrom(tmp)
    best_n_estimators_value = (value[0][0] + 1) * 10
    best_max_depth_value = value[0][1] + 1
    best_max_features_value = value[0][2] + 1 + 3
    return best_n_estimators_value, best_max_depth_value, best_max_features_value


# 求目标函数的和，用于自然选择
def sum(fit_value):
    s = 0
    for i in fit_value:
        s += i
    return s


# 计算概率的累积，类似前缀和，用于自然选择
def presum(fit_value):
    s = 0.0
    tmp = []
    for i in range(len(fit_value)):
        s += fit_value[i]
        tmp.append(s)
    for i in range(len(fit_value)):
        fit_value[i] = tmp[i]


# 自然选择（轮盘赌算法实现）
def selection(pop, fit_value):
    new_fit_value = []
    # 计算每个适应值的概率
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    presum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法
    fitin = 0
    newin = 0
    newpop = pop
    # ms预先根据大小进行排序，逐个找到之前概率前缀和相应的位置，进行选择：
    while newin < pop_len:
        if ms[newin] < new_fit_value[fitin]:
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


# 个体染色体交叉遗传
def crossover(pop, copulation):
    plen = len(pop)
    for i in range(plen - 1):
        if random.random() < copulation:
            tmp1 = []
            tmp2 = []
            # 中间位置分割的单点交叉
            tmp1.extend(pop[i][0:5])
            tmp1.extend(pop[i + 1][5:10])
            tmp2.extend(pop[i + 1][0:5])
            tmp2.extend(pop[i][5:10])
            pop[i] = tmp1
            pop[i + 1] = tmp2


# 基因突变
def mutation(pop, variation):
    p = len(pop[0])
    for i in range(len(pop)):
        if random.random() < variation:
            mpoint = random.randint(0, p - 1)
            if pop[i][mpoint] == 1:
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


# 交叉验证的结果
def cross_score(n_estimators, max_depth, max_features):
    data = pd.read_csv(open("data/data.csv", "r"))
    X = data.values[:, :-1]
    Y = data["VCI"]
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
    )
    score = cross_val_score(rfc, X, Y, cv=5, scoring="accuracy")
    return score.mean()


# 求交叉验证的最好的值
def best_res(result):
    tmp = result[-1]
    m_score = tmp[0]
    r_n_estimators, r_max_depth, r_max_features = tmp[1], tmp[2], tmp[3]
    max_score = cross_score(tmp[1], tmp[2], tmp[3])
    i = len(result) - 2
    while i >= 0:
        if result[i][0] != m_score:
            break
        temp = result[i]
        t_score = cross_score(temp[1], temp[2], temp[3])
        if t_score > max_score:
            r_n_estimators = temp[1]
            r_max_depth = temp[2]
            r_max_features = temp[3]
            max_score = t_score
    return [m_score, max_score, r_n_estimators, r_max_depth, r_max_features]


if __name__ == "__main__":
    pop = code_chrom(populations, chrom_len)
    data = pd.read_csv(open("data.csv", "r"))
    # 划分训练集和测试集
    train, test = train_test_split(data, test_size=0.3, random_state=80)
    k_range = []
    for i in range(generations):
        k_range.append(i)
        print("第 %s 代：" % (i + 1))
        # 计算目标函数值
        obj_value = com_objvalue(pop, train, test)
        # 转化为适应度值
        fit_value = com_fitvalue(obj_value)
        # 选出最好的目标函数和个体
        [best_individual, best_fit] = get_best(pop, obj_value)
        # 求出每一代平均的目标函数值
        score_sum = 0
        for j in obj_value:
            score_sum = score_sum + j
        average_score = score_sum / populations
        avg_scores.append(average_score)
        tmp_n_estimator, tmp_max_depth, tmp_max_features = record_best(best_individual)
        max_scores.append([best_fit, tmp_n_estimator, tmp_max_depth, tmp_max_features])
        print(
            "best_individual: "
            + str(best_individual)
            + " "
            + "best_score: "
            + str(best_fit)
            + " "
            + "average_score:"
            + str(average_score)
        )
        # 自然选择
        selection(pop, fit_value)
        # 交叉遗传
        crossover(pop, copulation)
        # 基因突变
        mutation(pop, variation)
    # 画出遗传过程中的平均得分
    plt.xlabel("k_range")
    plt.ylabel("Accurancy")
    plt.plot(k_range, avg_scores, "ob-")
    plt.xlim(-1, generations + 1)
    plt.ylim(0.7, 0.85)
    plt.show()
    max_scores.sort()
    # 对于测试集得分最高的一批进行交叉验证，测试泛化能力，求交叉验证得分最高的
    res = best_res(max_scores)
    print(
        "最好结果：测试集评分：%s 交叉验证评分：%s 树的数量: %s 最大深度：%s 划分最大特征：%s"
        % (res[0], res[1], res[2], res[3], res[4])
    )
