import os
import pandas as pd
from tabulate import tabulate

filenames= os.listdir (".")

table = []

last_rows = [{
        'val_T_loss': 3.4140e-07,
        'val_dT_dn_loss': 0.0041,
        'val_T_mae': 4.6139e-04,
        'val_dT_dn_mae': 0.0403
},
{
        'val_T_loss': 1.4606e-06,
        'val_dT_dn_loss': 0.0073,
        'val_T_mae': 0.0010,
        'val_dT_dn_mae': 0.0550
},
{
        'val_T_loss': 3.2325e-07,
        'val_dT_dn_loss': 0.0035,
        'val_T_mae': 4.5497e-04,
        'val_dT_dn_mae': 0.0206
},
{
        'val_T_loss': 3.2528e-07,
        'val_dT_dn_loss': 5.3124e-04,
        'val_T_mae': 4.4779e-04,
        'val_dT_dn_mae': 0.0116
},
{
        'val_T_loss': 3.0268e-7,
        'val_dT_dn_loss': 4.7512e-04,
        'val_T_mae': 4.3742e-04,
        'val_dT_dn_mae': 0.0125
}
]

names = [
        'CCNNNGauss',
        'CCNNNGaussOpt',
        'CCNNTrig',
        'CCNNTrigOpt',
        'CNN'
]

# for name, last_row in zip(names, last_rows):
#         tab = [name,
#                 r"${:.4f}$".format(last_row['val_T_loss'] * 627.503),
#                 r"${:.4f}$".format(last_row['val_T_mae'] * 627.503),
#                 r"${:.4f}$".format(last_row['val_dT_dn_loss'] * 627.503),
#                 r"${:.4f}$".format(last_row['val_dT_dn_mae'] * 627.503)]

#         table.append(tab)

#         # print(losses_val.iloc[-1]['val_dT_dn_mae'] * 627.503)
# header = ["model","loss", "mean", 'loss', 'mean']
# print(tabulate(table, headers=header, tablefmt="latex_raw"))




for folder in filenames:
    if '.py' not in folder and 'Exp' in folder:
        losses_val = pd.read_csv(folder + '/losses_val.csv')
        last_row = losses_val.iloc[-1] * 627.503

        tab = [folder.split('_')[0],
                r"${:.4f}$".format(last_row['val_T_loss']),
                r"${:.4f}$".format(last_row['val_T_mae']),
                r"${:.4f}$".format(last_row['val_dT_dn_loss']),
                r"${:.4f}$".format(last_row['val_dT_dn_mae'])]

        table.append(tab)

        # print(losses_val.iloc[-1]['val_dT_dn_mae'] * 627.503)
header = ["model","loss", "mean", 'loss', 'mean']
print(tabulate(table, headers=header, tablefmt="latex_raw"))
