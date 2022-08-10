import numpy as np

from daguerreo import structures as st, evaluation as ev
import torch
from daguerreo.tests.test_structures import StructureCase, all_diff


def shd(A, B):
    return ev.count_accuracy(A, B)['shd']


def shd_2():
    shds = []
    for _ in range(100):
        d = torch.randint(1000, (1,)).item()
        c = StructureCase()
        c.setUp(d)
        eps = 1.e-5
        if all_diff(c.smap_str.theta, eps):
            c.smap_str.eval()
            i, j = torch.randint(d, (2,))
            if i != j:
                c.smap_str.theta.data[i] = c.smap_str.theta.data[j] - eps
                _, m1, _ = c.smap_str()

                c.smap_str.theta.data[i] = c.smap_str.theta.data[j] + eps

                _, m2, _ = c.smap_str()
                shds.append(shd(m1[0].numpy(), m2[0].numpy()))

    test_all_one = torch.all(torch.tensor(shds)==1)
    print(test_all_one)


def shd_general(KK):
    shds = []
    for _ in range(100):
        d = 2 + KK + torch.randint(1000, (1,)).item()
        c = StructureCase()
        c.setUp(d)
        eps = 1.e-6
        if all_diff(c.smap_str.theta, KK*eps):
            c.smap_str.eval()
            indices = torch.randint(d, (KK,))
            if len(np.unique(indices)) == len(indices):
                for k, idx in enumerate(indices[1:]):
                    c.smap_str.theta.data[idx] = c.smap_str.theta.data[indices[0]] - (k+1)*eps
                _, m1, _ = c.smap_str()

                for k, idx in enumerate(indices[1:]):
                    c.smap_str.theta.data[idx] = c.smap_str.theta.data[indices[0]] + (k+1)*eps

                _, m2, _ = c.smap_str()
                shds.append(shd(m1[0].numpy(), m2[0].numpy()))
                print(shds[-1])
    print(shds)
    test_max_SHD = torch.all(torch.tensor(shds)==np.sum(range(0, KK)))
    print(test_max_SHD)
    return test_max_SHD


def shd_topk():
    shds = []
    for _ in range(1):
        # d = torch.randint(1000, (1,)).item()
        d = 6
        c = StructureCase()
        c.setUp(d)
        if all_diff(c.smap_str.theta, 1.e-5):
            alphas, dags, _ = c.smax_str()
            for d1, d2 in zip(dags, dags[1:]):

                shds.append(shd(d1.numpy(), d2.numpy()))
                print(shds[-1])

    test_all_one = torch.all(torch.tensor(shds)==1)
    print(test_all_one)
    # mh... this is not 1 as I imagined... can we say something about this?


if __name__ == '__main__':
    # shd_2()
    print([shd_general(k) for k in range(2, 20)])
    shd_topk()
