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
        for line in log:
            if '=' not in line:
                continue
            dicts = {}
            for col in line.split(','):
                if '=' not in col:
                    continue
                k, v = col.split('=', 1)
                dicts[k.strip().split()[-1]] = v.strip()
            if not {'step', 'energy_rmse', 'forces_rmse'}.issubset(dicts):
                continue
            step.append(dicts['step'])
            energy_rmse.append(float(dicts['energy_rmse']))
            forces_rmse.append(float(dicts['forces_rmse']))
    tuples = {
            'step': step,
            'energy_rmse': energy_rmse,
            'forces_rmse': forces_rmse,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(f'{path}/{output_dir}/{save}.csv')
    return df


def plot_energy_rmse(df, i, dir, save):
    fig, ax = plt.subplots()
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['energy_rmse'], color=f'C{i}', label=dir)
    ax.set_ylabel('Binding energy RMSE')
    ax.tick_params(axis="y")
    plt.ylim([-0.05, 1.3])
    if save:
        fig.savefig(f'{path}/{output_dir}/{save}.png')
    # plt.show()
    return ax

def plot_forces_rmse(df, i, dir, save):
    fig, ax = plt.subplots()
    plt.xlabel('Number of step')
    ax.plot(df['step'], df['forces_rmse'], color=f'C{i}', label=dir)
    ax.set_ylabel('Forces RMSE')
    ax.tick_params(axis="y")
    plt.ylim([0.05, 0.35])
    if save:
        fig.savefig(f'{path}/{output_dir}/{save}_forces.png')
    # plt.show()
    return ax

if __name__ == '__main__':
    output_dir = 'output'
    name = f'{path}/{output_dir}/output.log'
    save = 'learning_curve'
    df = generate_csv(name, save)
    df = pd.read_csv(f'{path}/{output_dir}/{save}.csv', skiprows=[])
    print(df)
    plot_energy_rmse(df, 1, 'Binding energy learning curve', save)
    plot_forces_rmse(df, 1, 'Adsorbate forces learning curve', save)
