import matplotlib.pyplot as plt
from math import sqrt




def plot_residuals(y,yhat):

    residual = yhat-y
    b_resid = y.mean()-y


    plt.figure(figsize = (11,5))

    plt.subplot(121)
    plt.scatter(y,b_resid)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('Baseline Residuals')

    plt.subplot(122)
    plt.scatter(y, residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title('Linear Regression Residual')

    plt.show()


def regression_errors(y,yhat):
    resid_sq = (yhat-y)**2
    SSE = resid_sq.sum()
    ESS = sum(y.mean()-yhat)
    TSS = SSE + ESS
    MSE = SSE/len(y)
    RMSE = sqrt(MSE)

    print(f'SSE: {SSE: .0f}')
    print(f'ESS: {ESS: f}')
    print(f'TSS: {TSS: .0f}')
    print(f'MSE: {MSE: .0f}')
    print(f'RMSE: {RMSE: .0f}')
    return(SSE,ESS,TSS,MSE,RMSE)
