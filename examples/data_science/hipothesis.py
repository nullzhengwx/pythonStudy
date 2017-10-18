import math
import random

from examples.data_science.propobility import normal_cdf, inverse_normal_cdf


def normal_approximation_to_binomial(n, p) :
    """
    finds mu and sigma corresponding to a Binomial(n, p)
    在二项式的标准差为 q*(1-q) 的情况下
    :param n:
    :param p:
    :return:
    """
    mu = p * n
    sigma = math.sqrt(p * (1 -p) * n)
    return mu, sigma

# 正态cdf是一个变量在一个阈值以下的概率
normal_probability_below = normal_cdf

def normal_probability_above(lo, mu=0, sigma=1):
    """如果它不在阈值一下,就在阈值以上"""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo, hi, mu=0, sigma=1) :
    """如果它小于hi但不比lo小,那么它在区间之内"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo, hi, mu=0, sigma=1) :
    """如果不在区间之内, 那么就在区间之外"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability, mu=0, sigma=1) :
    """return the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1) :
    """returns the z for which P(Z > z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1) :
    """returns the symmetric (about the mean) bounds
    that contain the specified probability"""
    tail_probability = (1 - probability) / 2

    # 上界应有在它之上的tail_probability
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # 下界应有在它之下的tial_probability
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # 如果x大于均值,tail表示比x大多少
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # 如果x比均值小, tail表示比x小多少
        return 2 * normal_probability_below(x, mu, sigma)

def a_b_test_statistic(N_A, n_A, N_B, n_B) :
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / M)
    return p, sigma

""" Beta分布"""
def B(alpha, beta) :
    """ a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta) :
    if x < 0 or x > 1:      # [0, 1]之外没有权重
        return 0
    return x ** ( alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

""" testing """
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print(mu_0, sigma_0)

# 基于假设p是0.5时95%的边界
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print(lo, hi)

# 基于p = 0.55 的真实mu和sigma
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
print(mu_1, sigma_1)

# 第2类错误意味着我们没有拒绝原假设
# 这会在X仍然在最初的区间时发生
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability  # 因为均值已经向右移为550, 所以type_2会很少
print(power)

# 单边,所以不同于上面的hi
hi_1 = normal_upper_bound(0.95, mu_0, sigma_0)
print(hi_1)

type_2_probability = normal_probability_below(hi_1, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)

print(two_sided_p_value(529.5, mu_0, sigma_0))

""" simulation """
extreme_value_count = 0
for _ in range(100000) :
    num_heads = sum(1 if random.random() < 0.5 else 0
                    for _ in range(1000))
    if num_heads >= 530 or num_heads <= 470 :
        extreme_value_count +=1

print(extreme_value_count / 100000)