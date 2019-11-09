import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def real_vs_fake(real_data, fake_data, x_col, y_col, sample_size):
    colors = ('blue', 'red')
    groups = ('fake', 'real')

    real_data = real_data.sample(sample_size)
    fake_data = fake_data.sample(sample_size)

    fake_data_points = (fake_data[x_col], fake_data[y_col])
    real_data_points = (real_data[x_col], real_data[y_col])
    total_data = (fake_data_points, real_data_points)

    for total_data, color, group in zip(total_data, colors, groups):
        x, y = total_data
        plt.scatter(x, y, alpha = 0.2, c=color, edgecolors = 'none', s=30, label = group)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc=2)
    plt.show()


# def real_vs_fake(real_data, fake_data_list, x_col, y_col):
#     fig, axs = plt.subplots(5, sharex=False)
#     fig.suptitle('Real and Fake Data Comparison')

#     colors = ('blue', 'red')
#     groups = ('fake', 'real')
    
#     for pnum in range(len(fake_data_list)):
#         fake_data = fake_data_list[pnum]
#         fake_data_points = (fake_data[x_col], fake_data[y_col])
#         real_data_points = (real_data[x_col], real_data[y_col])
#         total_data = (fake_data_points, real_data_points)

#         for total_data, color, group in zip(total_data, colors, groups):
#             x, y = total_data
#             axs[pnum].scatter(x, y, alpha = 0.2, c=color, edgecolors='none', s=30, label=group)

#     plt.xlabel(x_col)
#     plt.ylabel(y_col)
#     plt.legend(loc=2)
#     plt.show()