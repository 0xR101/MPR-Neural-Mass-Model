from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

Delta, J, tau = 1.0, 15.0, 1.0
kappa = J/np.pi
base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
out = base/'mpr_3d_results'


def rhs(s, y, h):
    x, v = y
    return [Delta+2*x*v, v*v-x*x+kappa*x+h]


def equilibria(h):
    roots = np.roots([1, -kappa, -h, 0, -Delta**2/4])
    x = np.sort(roots.real[(abs(roots.imag) < 1e-9) & (roots.real > 0)])
    return np.column_stack((x, -Delta/(2*x)))


def matrix(y):
    x, v = y
    return np.array([[2*v, 2*x], [kappa-2*x, 2*v]])


def direction(y, mode):
    values, vectors = np.linalg.eig(matrix(y))
    i = np.argmax(values.real) if mode == 'unstable' else np.argmin(abs(values))
    d = vectors[:, i].real
    d = d/np.linalg.norm(d)
    return d if d[0] > 0 else -d


def case(name, h, equilibrium, epsilon, d, end):
    d = np.asarray(d, dtype=float)
    d = d/np.linalg.norm(d)
    s = np.linspace(0, end, int(round(end/0.01))+1)
    traces, errors = [], []
    for sign in [-1, 1]:
        initial = equilibrium+sign*epsilon*d
        solutions = []
        for rtol, atol, step in [(1e-9, 1e-11, 0.03), (1e-11, 1e-13, 0.015)]:
            sol = solve_ivp(rhs, (0, end), initial, args=(h,), t_eval=s,
                            method='RK45', rtol=rtol, atol=atol, max_step=step)
            if not sol.success or not np.isfinite(sol.y).all() or sol.y[0].min() < 0:
                raise RuntimeError('Integration failed: '+name)
            solutions.append(sol.y)
        errors.append(np.max(abs(solutions[0]-solutions[1]), axis=1).tolist())
        # Convert x to the physical firing-rate variable r, retaining v.
        traces.append(np.vstack((solutions[0][0]/(np.pi*tau), solutions[0][1])))
    residual = float(np.max(abs(np.asarray(rhs(0, equilibrium, h)))))
    assert residual < 1e-10
    eig = np.linalg.eigvals(matrix(equilibrium))
    record = dict(h=float(h), equilibrium_x_v=equilibrium.tolist(),
                  equilibrium_r_v=[float(equilibrium[0]/(np.pi*tau)), float(equilibrium[1])],
                  epsilon_x_v=epsilon, direction_x_v=d.tolist(), end_s=end,
                  eigenvalues_s=[[float(z.real), float(z.imag)] for z in eig],
                  equilibrium_residual=residual, refinement_errors_x_v=errors)
    np.savetxt(out/f'{name}.csv', np.column_stack((s, traces[0].T, traces[1].T)),
               delimiter=',', header='s,r_minus,v_minus,r_plus,v_plus', comments='')
    return dict(s=s, traces=traces, record=record)


colors = ['0.15', '#36648b']


def panel(ax, data, title, stop=None):
    s = data['s']
    keep = np.ones(s.shape, dtype=bool) if stop is None else s <= stop
    end = s[keep][-1]
    rstar, vstar = data['record']['equilibrium_r_v']
    ax.plot([0, end], [rstar, rstar], [vstar, vstar],
            color='0.6', ls='--', lw=0.85)
    for color, y in zip(colors, data['traces']):
        ax.plot(s[keep], y[0, keep], y[1, keep], color=color, lw=1.0)
        ax.scatter([0], [y[0, 0]], [y[1, 0]], color=color, s=10, depthshade=False)
    ax.set(xlim=(0, end), xlabel=r'$t/\tau$', ylabel=r'$r(t)$', zlabel=r'$v(t)$')
    ax.set_title(title, fontsize=10, pad=12)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.45, 1, 1))
    ax.tick_params(labelsize=8, pad=1, width=0.6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(MaxNLocator(nbins=3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        axis.pane.fill = False
        axis.pane.set_edgecolor('0.88')
        axis._axinfo['grid'].update(color='0.89', linewidth=0.45)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 7
    ax.zaxis.labelpad = 7


def finish(fig, name):
    handles = [Line2D([0], [0], color='0.6', ls='--', lw=0.85, label='Equilibrium'),
               Line2D([0], [0], color=colors[0], lw=1.1, label='Negative perturbation'),
               Line2D([0], [0], color=colors[1], lw=1.1, label='Positive perturbation')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.01, right=0.96, bottom=0.17, top=0.84, wspace=0.13)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'{name}.{ext}', dpi=600 if ext == 'png' else 150,
                    bbox_inches='tight', pad_inches=0.15)


def main():
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix',
                         'font.size':9, 'axes.linewidth':0.7,
                         'pdf.fonttype':42, 'svg.fonttype':'path'})
    eq = equilibria(-5)
    generic = [1, 0.5]
    data = {}
    data['node'] = case('node', -5, eq[0], 0.05, generic, 4)
    data['saddle'] = case('saddle', -5, eq[1], 0.001, direction(eq[1], 'unstable'), 25)
    data['focus'] = case('focus', -5, eq[-1], 0.05, generic, 20)
    fig = plt.figure(figsize=(11.4, 4.3))
    for i, (name, label) in enumerate([('node', 'Stable node'), ('saddle', 'Saddle'),
                                       ('focus', 'Stable focus')]):
        rstar, vstar = data[name]['record']['equilibrium_r_v']
        title = label+'\n'+rf'$r_*={rstar:.3f},\ v_*={vstar:.3f}$'
        panel(fig.add_subplot(1, 3, i+1, projection='3d'), data[name], title)
    fig.suptitle(r'$\Delta=1,\quad J=15,\quad h=-5,\quad \tau=1$', y=0.98, fontsize=11)
    finish(fig, 'mpr_3d_equilibria')

    roots = np.roots([4, -2*kappa, 0, 0, Delta**2])
    xf = roots.real[(abs(roots.imag) < 1e-10) & (roots.real > 0)]
    hf = -xf**2-3*Delta**2/(4*xf**2)
    i = np.argmax(hf)
    fold = np.array([xf[i], -Delta/(2*xf[i])])
    data['fold'] = case('fold', hf[i], fold, 0.02, direction(fold, 'zero'), 110)
    negative, positive = data['fold']['traces']

    print(f"Final time t/tau = {data['fold']['s'][-1]:.2f}")
    print(f"{'Trajectory':<24} {'Final r':>15} {'Final v':>15}")
    print(f"{'Negative perturbation':<24} "
          f"{negative[0, -1]:15.9f} {negative[1, -1]:15.9f}")
    print(f"{'Positive perturbation':<24} "
          f"{positive[0, -1]:15.9f} {positive[1, -1]:15.9f}")
    print(f"{'Saddle-node equilibrium':<24} "
          f"{fold[0]/(np.pi*tau):15.9f} {fold[1]:15.9f}")
    fig = plt.figure(figsize=(8.2, 4.5))
    panel(fig.add_subplot(1, 2, 1, projection='3d'), data['fold'], 'Saddle-node: local view', stop=25)
    panel(fig.add_subplot(1, 2, 2, projection='3d'), data['fold'], 'Saddle-node: longer-time evolution')
    fig.suptitle(rf'$h_{{\rm SN}}={hf[i]:.9f},\quad \tau=1$', y=0.98, fontsize=11)
    finish(fig, 'mpr_3d_saddle_node')

    xnf = kappa/2
    hnf = -kappa**2/4-Delta**2/kappa**2
    settings = [('nf_node', hnf-0.0002, equilibria(hnf-0.0002)[-1], 'Stable node'),
                ('nf_boundary', hnf, np.array([xnf,-Delta/(2*xnf)]), 'Node-focus point'),
                ('nf_focus', hnf+0.05, equilibria(hnf+0.05)[-1], 'Stable focus')]
    fig = plt.figure(figsize=(12.2, 4.6))
    for i, (name, h, equilibrium, title) in enumerate(settings):
        data[name] = case(name, h, equilibrium, 0.0001, generic, 25)
        panel(fig.add_subplot(1, 3, i+1, projection='3d'), data[name],
              title+'\n'+rf'$h={h:.6f}$')
    finish(fig, 'mpr_3d_node_focus')
    metadata = dict(Delta=Delta, J=J, kappa=kappa, tau=tau,
                    units='tau=1 arbitrary time unit; r is inverse of that time unit',
                    cases={name: d['record'] for name, d in data.items()})
    (out/'case_details.json').write_text(json.dumps(metadata, indent=2)+'\n')
    error = max(np.max(d['record']['refinement_errors_x_v']) for d in data.values())
    print(f'Largest absolute refinement difference in x or v: {error:.3e}')
    print('Figures and data:', out)
    plt.show()


if __name__ == '__main__':
    main()
