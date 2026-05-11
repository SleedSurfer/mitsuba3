import numpy as np
import matplotlib.pyplot as plt


def plot_nimbus_logic_clear(R=1.0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # --- Horný graf: Geometria ---
    normals = [np.array([1.0, 0.0]), np.array([0.5, 0.866]), np.array([-0.5, 0.866])]
    colors = ['#ff4444', '#44aa44', '#4444ff']

    # Vykreslenie hexagonu (prienik)
    angles = np.linspace(0, 2 * np.pi, 7)
    v_rad = R / np.cos(np.radians(30))
    ax1.fill(v_rad * np.cos(angles + np.radians(30)), v_rad * np.sin(angles + np.radians(30)),
             color='#eeeeee', ec='k', lw=2, label='Kryštál (Prienik všetkých slabov)')

    # Lúč
    origin = np.array([-2.2, -0.6])
    direction = np.array([1.0, 0.35])
    direction /= np.linalg.norm(direction)

    t_vals = np.linspace(0, 4.5, 500)
    ray_pts = origin + t_vals[:, None] * direction
    ax1.plot(ray_pts[:, 0], ray_pts[:, 1], 'k-', alpha=0.2, label='Trajektória lúča')

    # Výpočet intervalov pre každý slab
    intervals = []
    for n in normals:
        denom = np.dot(direction, n)
        p_dot_n = np.dot(origin, n)
        t1, t2 = (R - p_dot_n) / denom, (-R - p_dot_n) / denom
        intervals.append(sorted([t1, t2]))

    # --- Dolný graf: Logika dr.maximum / dr.minimum ---
    ax2.set_title("Logika výpočtu priesečníka (Slab Intervals)")

    for i, (tn, tf) in enumerate(intervals):
        # Vykreslenie intervalu pre každý slab
        ax2.plot([tn, tf], [i, i], color=colors[i], lw=8, alpha=0.6, label=f"Slab {i + 1} interval")
        ax2.text(tn, i + 0.2, f"tn{i}", color=colors[i], fontweight='bold')
        ax2.text(tf, i + 0.2, f"tf{i}", color=colors[i], fontweight='bold')

    # Výsledné t_entry a t_exit
    t_entry = max([it[0] for it in intervals])
    t_exit = min([it[1] for it in intervals])

    # Zvýraznenie výsledného prieniku
    ax2.axvspan(t_entry, t_exit, color='#ff8800', alpha=0.3, label='Výsledný prienik (hit_mask)')
    ax2.axvline(t_entry, color='red', ls='--', lw=2)
    ax2.axvline(t_exit, color='blue', ls='--', lw=2)

    # Synchronizácia s horným grafom
    ax1.scatter(*(origin + direction * t_entry), color='red', s=100, zorder=10, label='t_entry (Vstup)')
    ax1.scatter(*(origin + direction * t_exit), color='blue', s=100, zorder=10, label='t_exit (Výstup)')
    ax1.plot([(origin + direction * t_entry)[0], (origin + direction * t_exit)[0]],
             [(origin + direction * t_entry)[1], (origin + direction * t_exit)[1]],
             color='#ff8800', lw=5, zorder=5)

    # Formátovanie
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.set_title("Geometrická reprezentácia priesečníka")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(['Slab 1', 'Slab 2', 'Slab 3'])
    ax2.set_xlabel("Parameter t (vzdialenosť pozdĺž lúča)")
    ax2.legend(loc='lower right', fontsize='x-small')

    plt.tight_layout()
    plt.savefig("slab_method.pdf")


plot_nimbus_logic_clear()