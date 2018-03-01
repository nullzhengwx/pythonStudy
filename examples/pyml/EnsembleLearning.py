# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt

# 集成分类器的出错率,二项分布
# 在假设分类器有相同的出错率,每个分类器都是独立的,且出错率不相关.
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

"""  training datas """

print(ensemble_error(n_classifier=11, error=0.25))

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]

plt.plot(error_range,
         ens_errors,
         label='Ensemble error',
         linewidth=2)

plt.plot(error_range,
         error_range,
         linestyle='--',
         label='Base error',
         linewidth=2)

plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

print(np.argmax(np.bincount([0, 0, 1],
          weights=[0.2, 0.2, 0.6])))