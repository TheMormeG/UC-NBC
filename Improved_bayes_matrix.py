"""
Created on：2024-05-12
Author:Wang Rukuan[TheMorme
Email: themorme@foxmail.com
Description: User Category-aware Naive Bayes Classifier
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import KFold
import warnings
from scipy.sparse import csr_matrix
import itertools
import itertools
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties

warnings.filterwarnings('ignore')  # 忽略警告

font_path = r'C:\Windows\Fonts\simsun.ttc'
chinese_font = FontProperties(fname=font_path, size=16)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12  # 全局默认字体大小


def sim_u_u_sparse(sparse_matrix_raw):  # 计算u与s相似度, 公式为sim_{us}
    average_user_rating = np.round(sparse_matrix_raw.sum(axis=1).A1 / np.diff(sparse_matrix_raw.indptr), 8) + 1e-10  # 取出每行非零值求mean,添加一个极小值防止sign时出现0
    average_data = np.repeat(average_user_rating, np.diff(sparse_matrix_raw.indptr))
    average_matrix = csr_matrix((average_data, sparse_matrix_raw.indices, sparse_matrix_raw.indptr), shape=sparse_matrix_raw.shape)
    matrix_diff = sparse_matrix_raw - average_matrix  # 均分矩阵-原矩阵
    matrix_diff_sign_data = np.sign(matrix_diff.data)  # 符号函数来判断是否positive还是negative
    matrix_diff_sign = csr_matrix((matrix_diff_sign_data, matrix_diff.indices, matrix_diff.indptr), shape=matrix_diff.shape)
    matrix_diff_sign.data = np.nan_to_num(matrix_diff_sign.data)  # 这里把nan值填充为0，因为之后的矩阵相乘时不会影响结果
    matrix_sim_1 = csr_matrix.dot(matrix_diff_sign, matrix_diff_sign.T)  # sign负号直接相乘，此时等到的结果是sum(相同) - sum(不同的)
    matrix_sim_abs = csr_matrix.dot(np.abs(matrix_diff_sign), np.abs(matrix_diff_sign).T)  # abs后直接相乘，此时等到的结果是sum(相同) + sum(不同的)
    matrix_sim_sum = matrix_sim_1 + (matrix_sim_abs - matrix_sim_1) / 2  # 此时可以用此方法仅留下相同的
    # matrix_count=0时说明两个user之间没有相同的评分item，因此需要进行处理，因其分子必为0，只需要将分母改成!=0即可
    matrix_sim_count = np.where(matrix_sim_abs.toarray() == 0, 1, matrix_sim_abs.toarray())
    matrix_sim = matrix_sim_sum / matrix_sim_count
    np.fill_diagonal(matrix_sim, 0)  # 将对角线元素（自身相似度）置为0
    matrix_sim = csr_matrix(matrix_sim)
    return matrix_sim


def sim_u_v_sparse(sparse_matrix_raw, matrix_similarity):  # 计算u与v的邻居的相似度平均值
    matrix_rating_mask = csr_matrix(
        (np.ones_like(sparse_matrix_raw.data), sparse_matrix_raw.indices, sparse_matrix_raw.indptr),
        shape=sparse_matrix_raw.shape)  # 将相似矩阵中的值全部变成1，方便矩阵直接相乘即相加
    matrix_similarity_mask = csr_matrix(np.ones(matrix_similarity.shape))  # 将相似矩阵中的值全部变成1，方便矩阵直接相乘即相加
    matrix_similarity_mask.setdiag(0)
    matrix_u_v_sum = csr_matrix.dot(matrix_similarity, matrix_rating_mask)  # 两个矩阵相乘，0值说明user与item的所有邻居的相似度都是0

    matrix_u_v_count = csr_matrix.dot(matrix_similarity_mask, matrix_rating_mask)
    # matrix_u_v_count=0时说明user与item所有邻居相似度都是0，因此其分子也是0，只需要将分母改成!=0即
    matrix_u_v_count = matrix_u_v_count.todense()
    matrix_u_v_count = np.where(matrix_u_v_count == 0, 1, matrix_u_v_count)
    matrix_u_v_sum = matrix_u_v_sum.todense()
    matrix_u_v = matrix_u_v_sum / matrix_u_v_count
    matrix_u_v = csr_matrix(matrix_u_v)
    return matrix_u_v


def data_rolling(df_raw, alpha):  # 按照窗口大小和步长对数据进行滚动
    df_raw += alpha  # 拉普拉斯平滑系数
    window_size = 3 if df_raw.shape[-1] // 25 < 3 else df_raw.shape[-1] // 25   # 窗口大小
    if isinstance(df_raw, pd.Series):
        df_raw = df_raw.rolling(window=window_size, min_periods=1).mean()  # 滚动平均
    else:
        df_raw = df_raw.rolling(window=window_size, min_periods=1, axis=1).mean()  # 滚动平均
    return df_raw


def rating_to_sparse_matrix(df_raw):  # 将评分矩阵转化为稀疏矩阵
    row = df_raw['user_id'].values
    col = df_raw['item_id'].values
    data = df_raw['rating'].values
    matrix = csr_matrix((data, (row, col)), shape=(df_raw['user_id'].max() + 1, df_raw['item_id'].max() + 1))
    return matrix


def progress_print(content):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {content}')


def classify_user(avg_rating, var_rating):
    global miu_list, var_list
    if avg_rating <= miu_list[1]:
        if var_rating <= var_list[1]:
            return 1
        elif var_rating <= var_list[2]:
            return 4
        else:
            return 7
    elif avg_rating <= miu_list[2]:
        if var_rating <= var_list[1]:
            return 2
        elif var_rating <= var_list[2]:
            return 5
        else:
            return 8
    else:
        if var_rating <= var_list[1]:
            return 3
        elif var_rating <= var_list[2]:
            return 6
        else:
            return 9


start_time = time.time()  # 开始计时
progress_print('程序开始运行')
dataset_name = 'movielens_100k'  # 选择数据集 movielens_100k、ciao、movielens_1m
if dataset_name == 'movielens_100k':
    df = pd.read_csv(r'.\UC-NBC\movielens-100k\u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
elif dataset_name == 'movielens_1m':
    df = pd.read_csv(r'.\UC-NBC\movielens-1m\ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
elif dataset_name == 'ciao':
    data = loadmat(r'.\UC-NBC\ciao_rating.mat')
    df = pd.DataFrame(data['rating'], columns=['user_id', 'item_id', 'cat_id', 'rating', 'helpfulness'])
    df = df[df['rating'] > 0]  # 剔除 rating 为 0 的情况
else:
    raise ValueError("未知数据集")
progress_print(f'{dataset_name}数据读取完毕')

df_rating = df.copy()  # 选择movielens-100k数据集
if 'timestamp' in df_rating.columns:
    df_rating = df_rating.sort_values(by=['timestamp']).reindex()  # 按照时间戳升序排序
else:
    df_rating = df_rating.reindex()

df_rating = df_rating[['user_id', 'item_id', 'rating']]  # 去掉timestamp
df_rating = df_rating.drop_duplicates(subset=['user_id', 'item_id'], keep='last').reindex()  # 去除重复数据

df_rating['user_id'] = df_rating['user_id'].factorize()[0] + 1  # 因user_id从1开始，因此需要+1
df_rating['item_id'] = df_rating['item_id'].factorize()[0] + 1  # 因item_id从1开始，因此需要+1

full_user_id = np.arange(1, df_rating['user_id'].max() + 1)  # 所有用户id
full_item_id = np.arange(1, df_rating['item_id'].max() + 1)  # 所有物品id

miu_list = [1, 2.5, 4, 5]
var_list = [0, 1.5, 3.5, 5]

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证
k = 1
lam = 0.1 if dataset_name == 'ciao' else 0.66  # 正则化系数
mae_each_fold = {'NB': [], 'INB': []}
for train_index, test_index in kf.split(df_rating):
    train_df = df_rating.iloc[train_index]  # 训练集
    test_df = df_rating.iloc[test_index]  # 测试集

    progress_print(f'Fold {k}开始处理相似矩阵')
    sparse_matrix = rating_to_sparse_matrix(train_df)  # 训练集转化为稀疏矩阵
    matrix_u_u_sim = sim_u_u_sparse(sparse_matrix)  # 计算用户之间的相似度
    matrix_user_item_sim = sim_u_v_sparse(sparse_matrix, matrix_u_u_sim)  # 计算用户与物品之间的相似度)
    progress_print(f'Fold {k}相似矩阵处理完成')
    # 对原数据添加需要的东西，包括user评价数量、user均分、item被评价数量、item均分、diff_rating
    # train_df['user_count'] = train_df.groupby('user_id')['user_id'].transform('count')
    train_df['user_avg_rating'] = train_df.groupby('user_id')['rating'].transform('mean')
    train_df['user_var_rating'] = train_df.groupby('user_id')['rating'].transform('var')

    train_df['user_cat'] = train_df.apply(lambda row: classify_user(row['user_avg_rating'], row['user_var_rating']), axis=1)

    train_df['item_degree'] = train_df.groupby('item_id')['item_id'].transform('count')
    train_df['item_avg_rating'] = train_df.groupby('item_id')['rating'].transform('mean')
    train_df['rating_diff'] = round(train_df['rating'] - train_df['user_avg_rating'], 1)
    train_df['sim_user_item'] = train_df.apply(lambda row: matrix_user_item_sim[int(row['user_id']) - 1, int(row['item_id']) - 1], axis=1)
    train_df['sim_user_item'] = train_df['sim_user_item'].fillna(0)  # 存在nan值，需要填充为0
    # 对user关于avg和var进行K-means聚类，得到user的分类并添加到train_df中
    # kmeans = KMeans(n_clusters=2, init=np.array([[3, 4.5], [1.5, 3.5]]), n_init=1, max_iter=1, tol=0.0001)
    # kmeans = KMeans(n_clusters=2)
    # user_kmeans = kmeans.fit_predict(train_df[['user_avg_rating', 'user_var_rating']].fillna(0))
    # train_df['user_kmeans'] = user_kmeans

    # 将train中的数据转化为字典，方便后续提供给test使用
    item_id_to_value = train_df.set_index('item_id')[['item_avg_rating', 'item_degree']].to_dict()
    user_id_to_value = train_df.set_index('user_id')[['user_avg_rating', 'user_var_rating']].to_dict()

    test_df['user_avg_rating'] = test_df['user_id'].map(user_id_to_value['user_avg_rating']).fillna(train_df['rating'].mean())
    # test_df['rating_diff'] = round(test_df['rating'] - test_df['user_avg_rating'], 1)

    # 以下将部分数据置入对应的区间,先创建需要bins
    bins_item_degree = np.round(np.arange(0, train_df['item_degree'].max()+5, 5), 0)
    bins_item_avg_rating = np.round(np.arange(0, train_df['item_avg_rating'].max()+0.1, 0.1), 1)
    bins_sim_u_v = np.round(np.arange(0, 1+0.2, 0.1), 1)

    # 再进行cut操作
    train_df['item_id_degree_cut'] = pd.cut(train_df['item_degree'], bins=bins_item_degree)
    train_df['item_id_avg_r_cut'] = pd.cut(train_df['item_avg_rating'], bins=bins_item_avg_rating)
    train_df['sim_uv_cut'] = pd.cut(train_df['sim_user_item'], bins=bins_sim_u_v, right=False)  # 最小是0，最大<1，因此左闭右开

    full_diff_index = pd.Index(np.round(np.linspace(-4, 4, 81), 1))
    C_len = len(full_diff_index)
    # 狄利克雷分布下的先验概率
    prior_C = df = pd.DataFrame(0, index=range(1, 10), columns=full_diff_index)

    for user_cat in range(1, 10):
        train = train_df[train_df['user_cat'] == user_cat]
        r = train['rating'].value_counts(normalize=True).sort_index()
        u_bar = round(train['user_avg_rating'], 1).value_counts().\
            reindex(np.round(np.linspace(1, 5, 41), 1), fill_value=0).sort_index()
        u_bar = data_rolling(u_bar, 10)  # 平滑处理
        u_bar = u_bar / u_bar.sum()  # 归一化后验概率密度
        for i in u_bar.index:
            for j in r.index:
                prior_C.loc[user_cat, round(j-i, 1)] += u_bar.loc[i] * r.loc[j] * 41 * 5

    # 常规概率分布
    C = train_df['rating_diff'].value_counts(normalize=True).sort_index().reindex(full_diff_index, fill_value=0)
    C = C / C.sum()  # 归一化后验概率密度

    # 计算条件概率：P(R^v|C_{uv})，需要考虑拉普拉斯平滑
    df_c_uv_r_v = pd.pivot_table(train_df, values='rating', index='rating_diff', columns='item_id_avg_r_cut', aggfunc='count')
    df_c_uv_r_v = df_c_uv_r_v.reindex(full_diff_index).fillna(0)  # 配合狄利克雷补全索引使用
    # df_c_uv_r_v += 20  # 实现拉普拉斯平滑
    df_c_uv_r_v = data_rolling(df_c_uv_r_v, 10)
    df_p_c_uv_r_v = df_c_uv_r_v.div(df_c_uv_r_v.sum(axis=1), axis=0)  # 每个值除以行总和，即可得频率（概率）

    # 计算条件概率：P(K^v|C_{uv})，需要考虑拉普拉斯平滑
    df_c_uv_k_v = pd.pivot_table(train_df, values='rating', index='rating_diff', columns='item_id_degree_cut', aggfunc='count')
    df_c_uv_k_v = df_c_uv_k_v.reindex(full_diff_index).fillna(0)  # 配合狄利克雷补全索引使用
    # df_c_uv_k_v += 10  # 实现拉普拉斯平滑
    df_c_uv_k_v = data_rolling(df_c_uv_k_v, 20)
    df_p_c_uv_k_v = df_c_uv_k_v.div(df_c_uv_k_v.sum(axis=1), axis=0)  # 每个值除以行总和，即可得频率（概率）

    # 计算条件概率：P(S^uv|C_{uv})，需要考虑拉普拉斯平滑
    df_c_uv_s_uv = pd.pivot_table(train_df, values='rating', index='rating_diff', columns='sim_uv_cut', aggfunc='count')
    df_c_uv_s_uv = df_c_uv_s_uv.reindex(full_diff_index).fillna(0)  # 配合狄利克雷补全索引使用
    # df_c_uv_s_uv += 10  # 实现拉普拉斯平滑
    df_c_uv_s_uv = data_rolling(df_c_uv_s_uv, 10)
    df_p_c_uv_s_uv = df_c_uv_s_uv.div(df_c_uv_s_uv.sum(axis=1), axis=0)  # 每个值除以行总和，即可得频率（概率）

    # 计算R_v概率，需要考虑拉普拉斯平滑
    # R_v = train_df['item_id_avg_r_cut'].value_counts().sort_index() + 1
    # R_v = R_v / R_v.sum()  # 可以直接通过R_v[value]进行获取值

    # 计算K_v概率，需要考虑拉普拉斯平滑
    # K_v = train_df['item_id_degree_cut'].value_counts().sort_index() + 1
    # K_v = K_v / K_v.sum()  # 可以直接通过K_v[value]进行获取值

    # 计算S_uv概率，需要考虑拉普拉斯平滑
    # S_uv = train_df['sim_uv_cut'].value_counts().sort_index() + 1
    # S_uv = S_uv / S_uv.sum()  # 可以直接通过K_v[value]进行获取值

    # 计算条件概率：P(S^{uv}}|C_{uv},R_v,K_v)，需要考虑拉普拉斯平滑
    df_s_uv_3 = pd.pivot_table(train_df, values='rating', index=['rating_diff', 'item_id_avg_r_cut', 'item_id_degree_cut'], columns='sim_uv_cut', aggfunc='count')
    df_s_uv_3_new_multi_index = pd.MultiIndex.from_product([full_diff_index, df_s_uv_3.index.levels[1], df_s_uv_3.index.levels[2]], names=df_s_uv_3.index.names)
    df_s_uv_3 = df_s_uv_3.reindex(df_s_uv_3_new_multi_index).fillna(0)
    df_s_uv_3 += 10  # 实现拉普拉斯平滑
    # df_s_uv_3 = data_rolling(df_s_uv_3, 10)
    df_p_s_uv_3 = df_s_uv_3.div(df_s_uv_3.sum(axis=1), axis=0)  # 每个值除以行总和，即可得频率（概率）

    # # 计算条件概率：P(S^{uv}}|R_v,K_v)，需要考虑拉普拉斯平滑
    # df_s_uv_2 = pd.pivot_table(train_df, values='rating', index=['item_id_avg_r_cut', 'item_id_degree_cut'], columns='sim_uv_cut', aggfunc='count')
    # # df_s_uv_2 += 5  # 实现拉普拉斯平滑
    # df_s_uv_2 = data_rolling(df_s_uv_2, 5)
    # df_s_uv_2 = df_s_uv_2.div(df_s_uv_2.sum(axis=0), axis=1)  # 每个值除以行总和，即可得频率（概率）

    # 以下是超参数的设定
    r_alpha = round(train_df[['item_avg_rating']].quantile(0.25).values[0], 1)  # 25%分位数
    r_beta = round(train_df[['item_avg_rating']].quantile(0.75).values[0], 1)  # 75%分位数

    batch_size = 512  # 批处理大小
    epoch = int(np.ceil(test_df.__len__() / batch_size))  # 预测轮数
    C_uv_value = np.array(full_diff_index)  # 为了提高运行效率，通过向量直接求解
    real_value = np.array(test_df['rating'])  # 真实评分值
    real_avg_value = np.array(test_df['user_avg_rating'])  # 真实均值

    pred_diff_improved = np.array([])
    pred_diff_ful_indep = np.array([])
    pred_diff_orig = np.array([])
    hit_count = 0

    # 对后验数据进行向量化，使其能够与先验直接相加：包括group by、reindex、对应user顺序排序
    for i in tqdm(range(epoch), desc=f'Fold {k}'):
        # 随机选择batch_size个样本进行训练
        test_data = test_df.iloc[i * batch_size: (i + 1) * batch_size][['user_id', 'item_id']].reset_index(drop=True)
        C_tiled = np.tile(np.array(C).reshape(-1, 1), (1, test_data.shape[0]))  # 常规概率分布

        # 获取需要的信息，包括商品的平均分和度数，用户的相似度矩阵，用户的平均分和方差
        item_avg_rating = test_data['item_id'].map(item_id_to_value['item_avg_rating']).fillna(train_df['item_avg_rating'].mean()).rename('item_avg_rating')
        item_degree = test_data['item_id'].map(item_id_to_value['item_degree']).fillna(1).rename('item_degree')

        sim_user_item = test_data.apply(lambda row: matrix_user_item_sim[int(row['user_id']) - 1, int(row['item_id']) - 1], axis=1)
        sim_user_item = sim_user_item.fillna(0)

        # 求用户分类及其对应的先验概率分布
        user_avg_rating = test_data['user_id'].map(user_id_to_value['user_avg_rating']).fillna(train_df['user_avg_rating'].mean()).rename('user_avg_rating')
        user_var_rating = test_data['user_id'].map(user_id_to_value['user_var_rating']).fillna(train_df['user_var_rating'].mean()).rename('user_var_rating')
        X = np.column_stack((user_avg_rating, user_var_rating))
        categories = [classify_user(row[0], row[1]) for row in X]
        C_uv_c = pd.DataFrame([prior_C.loc[i] for i in categories]).T
        C_uv_c = np.array(C_uv_c)
        # user自身rating数据，用于后验更新
        post_rating_diff = train_df[(train_df['user_id'].isin(test_data['user_id']))][['user_id', 'rating_diff']]
        # 对后验数据进行向量化，使其能够与先验直接相加：包括group by、reindex、对应user顺序排序
        # k_user_rating_diff = np.log(post_rating_diff.groupby(['rating_diff', 'user_id']).size().unstack(fill_value=0)+1)
        k_user_rating_diff = post_rating_diff.groupby(['rating_diff', 'user_id']).size().unstack(fill_value=0)

        k_user_rating_diff = k_user_rating_diff.reindex(columns=test_data['user_id'].drop_duplicates(), fill_value=0)  # 针对训练集中未出现过的user的rating数据
        k_user_rating_diff = k_user_rating_diff.reindex(full_diff_index, fill_value=0)[test_data['user_id']]
        k_user_rating_diff = np.array(k_user_rating_diff)  # 转为numpy数组
        # C_uv_c += k_user_rating_diff  # 更新后验概率
        C_uv_c = lam * C_uv_c + (1 - lam) * k_user_rating_diff  # 加权平均
        C_uv_c /= C_uv_c.sum(axis=0).reshape(1, -1)  # 归一化

        # 计算条件概率：P(X|C_{uv})
        C_uv_c_r = np.array(df_p_c_uv_r_v.loc[C_uv_value, item_avg_rating])
        C_uv_c_k = np.array(df_p_c_uv_k_v.loc[C_uv_value, item_degree])
        C_uv_c_s = np.array(df_p_c_uv_s_uv.loc[C_uv_value, sim_user_item])

        # 计算条件概率：P(S^{uv}|C_{uv},R_v,K_v) 和 P(S^{uv}|R_v,K_v)
        C_uv_s_3 = np.array([])
        C_uv_s_2 = np.array([])
        for j in range(len(sim_user_item)):
            C_uv_s_3 = np.append(C_uv_s_3, np.array(df_p_s_uv_3.loc[(slice(None), item_avg_rating.iloc[j], item_degree.iloc[j]), sim_user_item.iloc[j]]))
            # C_uv_s_2 = np.append(C_uv_s_2, np.array(df_s_uv_2.loc[(item_avg_rating.iloc[j],  item_degree.iloc[j]), sim_user_item.iloc[j]]))
        C_uv_s_3 = C_uv_s_3.reshape(-1, C_len).T

        # 计算概率：P(R_v), P(K_v), P(S_uv)
        # C_uv_r = np.array(R_v[item_avg_rating])
        # C_uv_k = np.array(K_v[item_degree])
        # C_uv_s = np.array(S_uv[sim_user_item])

        # 分别计算优化后的贝叶斯公式和朴素贝叶斯
        value_improved = (C_uv_c * C_uv_c_r * C_uv_c_k * C_uv_s_3)  # / np.tile((C_uv_r * C_uv_k * C_uv_s_2).reshape(1, -1), (C_len, 1))
        value_full_independence = (C_uv_c * C_uv_c_r * C_uv_c_k * C_uv_c_s)  # / np.tile((C_uv_r * C_uv_k * C_uv_s_2).reshape(1, -1), (C_len, 1))
        value_original = (C_tiled * C_uv_c_r * C_uv_c_k * C_uv_c_s)  # / np.tile((C_uv_r * C_uv_k * C_uv_s).reshape(1, -1), (C_len, 1))

        # 对于高/低评分的物品，使用改进的概率计算
        value_improved = np.where((item_avg_rating >= r_beta) | (item_avg_rating <= r_alpha), value_improved, value_full_independence)

        # 向量方法时的误差函数求解
        for v in [value_improved, value_original]:
            pred_value = C.index[np.argmax(v, axis=0)].values

            if v is value_improved:
                pred_diff_improved = np.append(pred_diff_improved, pred_value)
            else:
                pred_diff_orig = np.append(pred_diff_orig, pred_value)

    # 计算预测值
    pred_val_improved = np.round(real_avg_value + pred_diff_improved, 1)
    pred_val_orig = np.round(real_avg_value + pred_diff_orig, 1)

    # 限制预测值范围
    pred_val_improved = np.clip(pred_val_improved, 1, 5)
    pred_val_orig = np.clip(pred_val_orig, 1, 5)
    # 创建错误值的 DataFrame
    df_err_improved = pd.DataFrame([round(x, 1) for x in abs(pred_val_improved - real_value)])
    df_err_orig = pd.DataFrame([round(x, 1) for x in abs(pred_val_orig - real_value)])
    # 计算错误值的平均绝对误差MAE
    df_err_improved_mae = round(df_err_improved.mean()[0], 4)
    df_err_orig_mae = round(df_err_orig.mean()[0], 4)

    mae_each_fold['NB'].append(df_err_orig_mae)
    mae_each_fold['INB'].append(df_err_improved_mae)

    k += 1
    break

progress_print(f"程序运行完毕，总耗时{time.time() - start_time:.2f}秒")

for metric in [mae_each_fold]:
    for key, values in metric.items():
        avg = sum(values) / len(values)
        values_str = "\t".join(map(str, values))
        print(f"{key}\t{values_str} {avg:.4f}")
    print()


# test_df['pred_rating'] = pred_val_orig
# df = test_df[['user_id', 'item_id', 'rating', 'pred_rating']]
# # # 计算Precision@N和Recall@N
# threshold = 4.0

#
# def precision_at_n_for_users(df, N, threshold):
#     grouped = df.groupby('user_id')
#     precision_list = []
#
#     for user, group in grouped:
#         user_item_count = len(group)
#         current_N = min(N, user_item_count)
#         group_sorted = group.sort_values(by='pred_rating', ascending=False).head(current_N)
#         relevant_items = group_sorted[group_sorted['rating'] >= threshold]
#         precision_n = len(relevant_items) / current_N
#         precision_list.append(precision_n)
#
#     mean_precision = sum(precision_list) / len(precision_list)
#     return mean_precision
#
#
# def recall_at_n_for_users(df, N, threshold):
#     grouped = df.groupby('user_id')
#     recall_list = []
#
#     for user, group in grouped:
#         user_item_count = len(group)
#         current_N = min(N, user_item_count)
#         group_sorted = group.sort_values(by='pred_rating', ascending=False).head(current_N)
#         relevant_items_total = group[group['rating'] >= threshold]
#         relevant_items_in_top_n = group_sorted[group_sorted['rating'] >= threshold]
#         if len(relevant_items_total) > 0:
#             recall_n = len(relevant_items_in_top_n) / len(relevant_items_total)
#         else:
#             recall_n = 0
#         recall_list.append(recall_n)
#
#     mean_recall = sum(recall_list) / len(recall_list)
#     return mean_recall
#
#
# N_values = range(1, 21)
# recall_values = []
# precision_values = []
# f1_values = []
#
# for N in N_values:
#     recall_n = recall_at_n_for_users(df, N, threshold)
#     precision_n = precision_at_n_for_users(df, N, threshold)
#     recall_values.append(recall_n)
#     precision_values.append(precision_n)
#     if recall_n + precision_n > 0:
#         f1_n = 2 * (precision_n * recall_n) / (precision_n + recall_n)
#     else:
#         f1_n = 0
#     f1_values.append(f1_n)
#
# plt.figure(figsize=(10, 6))
# plt.plot(N_values, recall_values, label='Recall', marker='^', markersize=8)
# plt.plot(N_values, precision_values, label='Precision', marker='*', markersize=8)
# plt.plot(N_values, f1_values, label='F1 Score', marker='o', markersize=8)
# plt.xlabel(f'Top_N for {dataset_name}', fontsize=16)
# plt.ylabel('Value', fontsize=16)
# plt.xticks([1, 5, 10, 15, 20])
# plt.legend(loc='lower right', fontsize=15)
# plt.grid(False)
# plt.show()
#

