import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def mutual_info(rs, X):
    grs = []
    for r in rs:
        mi = np.mean(np.diag(np.add.outer(np.diag(X), np.diag(X))-X, k=r))
        grs.append(mi)
    return np.array(grs)

def corrs(rs, X):
    grs = []
    for r in rs:
        mi = np.mean(np.diag(X, k=r))
        grs.append(mi)
    return np.array(grs)

def p_neq(X):
    v = np.mean(np.diagonal(X, axis1=1, axis2=2), axis=0)
    return np.outer(v, v)

def p_01(X):
    rows = np.expand_dims(np.diagonal(X, axis1=1, axis2=2), axis=-1)
    ones_mask = rows*rows.transpose([0, 2, 1]) # ones_mask[i, j] = 1 if and only if (S_A)=diag(x)[i, i] and S_B=diag(x)[j, j] are 1
    twos_mask = (X==1).astype(int) + (X==2).astype(int) # twos_mask[i, j] = 1 if and only if, S_AB = x[i, j] > 0 (i.e. = 1, 2)
    #diagonal_mask = np.expand_dims(1-np.eye(*X.shape[1:], dtype=np.uint8), 0)
    return np.mean(ones_mask*twos_mask, axis=0) # mean of 1s where both masks hold is probability that both masks hold at each spot. 

def p_2(X):
    rows = np.expand_dims(np.diagonal(X, axis1=1, axis2=2), axis=-1)
    ones_mask = rows*rows.transpose([0, 2, 1]) # ones_mask[i, j] = 1 if and only if diag(x)[i, i] and diag(x)[j, j] are 1
    twos_mask = X==0 # twos_mask[i, j] = 1 if and only if, S_AB = x[i, j] = 0
    #diagonal_mask = np.expand_dims(1-np.eye(*X.shape[1:], dtype=np.uint8), 0)
    return np.mean(ones_mask*twos_mask, axis=0) # mean of 1s where both masks hold is probability that both masks hold at each spot. 

if __name__=='__main__':
    print('loading')
    x = np.load('data/SAB.npy')
    print('loaded')
    L = x.shape[1]
    a = 8

    X_trunc = x[:, L//a:(a-1)*L//a, L//a:(a-1)*L//a]
    two = p_2(X_trunc)
    zeroone = p_01(X_trunc)
    neq = p_neq(X_trunc)
    meaned = np.mean(X_trunc, axis=0)

    rs = np.arange(1, L//2)
    fig, ax = plt.subplots(1, 3, figsize=(10, 4), dpi=130)
    #ax = [ax]

    ty = corrs(rs, two)
    ny = corrs(rs, neq)
    zy = corrs(rs, zeroone)
    mi = mutual_info(rs, meaned)

    ax[0].plot(rs, ty, marker='.', label='two')

    ax[0].plot(rs, np.abs((zy-ny)), marker='.', label='zeroone-neq')

    ax[1].plot(rs, (zy-ny)/ty, marker='.', label='zeroone-neq/two')

    ax[2].plot(rs, mi, marker='.', label='mi')

    ax[0].set_yscale("symlog", linthresh=1e-10)
    ax[0].set_xscale("symlog")

    ax[1].set_xscale("symlog")
    ax[1].set_yscale("symlog")

    ax[2].set_xscale("log")
    ax[2].set_yscale("log")

    ax[0].set_xlabel('$r$')
    ax[1].set_xlabel('$r$')
    ax[2].set_xlabel('$r$')

    ax[0].set_ylabel('prob')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, dpi=130, figsize=(6, 4), sharey=True)
    ax[0].plot(rs, zy, label='zeroone')
    ax[1].plot(rs, ny, label='neq')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('$r$')
    ax[1].set_xlabel('$r$')
    ax[0].set_ylabel('prob')
    plt.show()
    

    fig, ax = plt.subplots(1, 3, dpi=130, figsize=(10, 4))
    ax[0].imshow(zeroone)
    ax[0].set_title('zeroone')
    ax[1].imshow(neq)
    ax[1].set_title('neq')
    ax[2].imshow(two)
    ax[2].set_title('two')
    plt.show()
