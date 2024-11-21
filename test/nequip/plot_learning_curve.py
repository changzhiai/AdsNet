import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
# mpl.use('TkAgg')

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def generate_csv(name, save):
    import pandas as pd
    step, energy_rmse, forces_rmse = [], [], []
    with open(name) as log:
        lines = log.readlines()
        i = 0
        for line in lines:
            i += 1
            if 'step=0,' in line:
                break
        print(i)
        for line in lines[i:]:
            if 'exiting' in line or 'stopping' in line:
                break
            cols = line.split(',')
            cols[1] = cols[1].split(' ')[-1]
            dicts = {}
            for col in cols[1:]:
                kv = col.split('=')
                k = kv[0].strip()
                v = kv[1].strip()
                dicts[k] = v
            # print(dicts)
            step.append(dicts['step'])
            energy_rmse.append(float(dicts['energy_rmse']))
            forces_rmse.append(float(dicts['forces_rmse']))
    tuples = {
            'step': step,
            'energy_rmse': energy_rmse,
            'forces_rmse': forces_rmse,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(f'{path}/{save}.csv')
    return df


def plot_energy_rmse(df, i, dir, save):
    fig, ax = plt.subplots()
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['energy_rmse'], color=f'C{i}', label=dir)
    ax.set_ylabel('Binding energy RMSE')
    ax.tick_params(axis="y")
    if save:
        fig.savefig(f'{path}/{save}.png')
    # plt.show()
    return ax

def plot_forces_rmse(df, i, dir, save):
    fig, ax = plt.subplots()
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['forces_rmse'], color=f'C{i}', label=dir)
    ax.set_ylabel('Forces RMSE')
    ax.tick_params(axis="y")
    if save:
        fig.savefig(f'{path}/{save}_forces.png')
    # plt.show()
    return ax

if __name__ == '__main__':
    name = f'{path}/model_output/printlog.txt'
    save = 'learning_curve'
    df = generate_csv(name, save)
    df = pd.read_csv(f'{path}/{save}.csv', skiprows=[])
    print(df)
    plot_energy_rmse(df, 1, 'Binding energy learning curve', save)
    # plot_forces_rmse(df, 1, 'Adsorbate forces learning curve', save)
