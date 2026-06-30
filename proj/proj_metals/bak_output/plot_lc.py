import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def generate_csv(name, save):
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
    df = pd.DataFrame({'step': step, 'energy_rmse': energy_rmse, 'forces_rmse': forces_rmse})
    df.to_csv(f'{path}/{output_dir}/{save}.csv')
    return df


def plot_individual(df, save):
    for col, ylabel, suffix, color in [
        ('energy_rmse', '$E_b$ RMSE (eV)',    'energy', 'C0'),
        ('forces_rmse', 'Forces RMSE (eV/Å)', 'forces', 'C1'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df['step'], df[col], color=color)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        if save:
            fname = f'{path}/{output_dir}/{save}_{suffix}.png'
            fig.savefig(fname, dpi=300)
            print(f'Saved figure to {fname}')
        plt.close()


def plot_separate(df, save):
    fig, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_e.plot(df['step'], df['energy_rmse'], color='C0')
    ax_e.set_ylabel('$E_b$ RMSE (eV)')
    ax_f.plot(df['step'], df['forces_rmse'], color='C1')
    ax_f.set_ylabel('Forces RMSE (eV/Å)')
    ax_f.set_xlabel('Step')
    plt.tight_layout()
    if save:
        fig.savefig(f'{path}/{output_dir}/{save}_separate.png', dpi=300)
        print(f'Saved figure to {path}/{output_dir}/{save}_separate.png')
    plt.close()


def plot_learning_curve(df, save):
    fig, ax_e = plt.subplots(figsize=(6, 4))
    ax_f = ax_e.twinx()
    l1, = ax_e.plot(df['step'], df['energy_rmse'], color='C0', label='Energy RMSE')
    l2, = ax_f.plot(df['step'], df['forces_rmse'], color='C1', label='Forces RMSE')
    ax_e.set_xlabel('Step')
    ax_e.set_ylabel('$E_b$ RMSE (eV)',    color='C0')
    ax_f.set_ylabel('Forces RMSE (eV/Å)', color='C1')
    ax_e.tick_params(axis='y', labelcolor='C0')
    ax_f.tick_params(axis='y', labelcolor='C1')
    ax_e.legend(handles=[l1, l2], loc='upper right', fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(f'{path}/{output_dir}/{save}.png', dpi=300)
        print(f'Saved figure to {path}/{output_dir}/{save}.png')
    plt.close()


if __name__ == '__main__':
    output_dir = 'output'
    name = f'{path}/{output_dir}/output.log'
    save = 'learning_curve'
    df = generate_csv(name, save)
    df = pd.read_csv(f'{path}/{output_dir}/{save}.csv')
    print(df)
    plot_individual(df, save)
    plot_separate(df, save)
    plot_learning_curve(df, save)

