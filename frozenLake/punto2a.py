
#   is_rainy=False  →  determinístico (prob=1.0 en dirección elegida)
#   is_rainy=True   →  estocástico    (prob=0.8 dirección elegida,


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SIMBOLOS_ACCION = {0: "↓", 1: "↑", 2: "→", 3: "←", 4: "P", 5: "D"}
NOMBRES_ACCION  = ["Sur", "Norte", "Este", "Oeste", "Recoger", "Dejar"]


def value_iteration(env, gamma=0.99, theta=1e-8):
  
    n_states  = env.observation_space.n   # 500
    n_actions = env.action_space.n        # 6

    V = np.zeros(n_states)
    deltas_historia = []

    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            v_actual = V[s]
            valores = [
                sum(prob * (reward + gamma * V[next_s])
                    for prob, next_s, reward, _ in env.P[s][a])
                for a in range(n_actions)
            ]
            V[s] = max(valores)
            delta = max(delta, abs(v_actual - V[s]))

        deltas_historia.append(delta)
        iteration += 1

        if delta < theta:
            print(f"  [VI] Convergió en {iteration} iteraciones  (delta final={delta:.2e})")
            break

    policy = np.array([
        np.argmax([
            sum(prob * (reward + gamma * V[next_s])
                for prob, next_s, reward, _ in env.P[s][a])
            for a in range(n_actions)
        ])
        for s in range(n_states)
    ], dtype=int)

    return V, policy, deltas_historia



def q_learning(env, n_episodes=10_000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.0005):

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    recompensas = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        estado, _ = env.reset()
        recompensa_total = 0
        done = False

        while not done:
            # ε-greedy
            if np.random.random() < epsilon:
                accion = env.action_space.sample()
            else:
                accion = int(np.argmax(Q[estado]))

            siguiente, reward, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated

            # Ecuación de actualización Q-Learning:
            # Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
            Q[estado, accion] += alpha * (
                reward + gamma * np.max(Q[siguiente]) - Q[estado, accion]
            )

            estado = siguiente
            recompensa_total += reward

        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        recompensas.append(recompensa_total)

        if (ep + 1) % 1000 == 0:
            media = np.mean(recompensas[-500:])
            print(f"  Ep {ep+1:>6} | ε={epsilon:.4f} | media ult-500: {media:.2f}")

    return Q, np.argmax(Q, axis=1), recompensas



def evaluar_politica(env, policy, n_episodios=1000):
    recompensas, pasos_lista = [], []
    for _ in range(n_episodios):
        estado, _ = env.reset()
        done = False
        r_total = 0
        pasos = 0
        while not done:
            accion = int(policy[estado])
            estado, reward, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            r_total += reward
            pasos += 1
        recompensas.append(r_total)
        pasos_lista.append(pasos)
    return {
        "media_recompensa": np.mean(recompensas),
        "std_recompensa":   np.std(recompensas),
        "media_pasos":      np.mean(pasos_lista),
        "exito_pct":        np.mean(np.array(recompensas) > 0) * 100,
        "recompensas":      recompensas,
    }


def mostrar_politica_ejemplo(env_wrapped, policy, titulo="Política", n_pasos=20):
   
    env_vis = gym.make("Taxi-v4", render_mode="ansi",
                       is_rainy=env_wrapped.unwrapped.is_rainy
                       if hasattr(env_wrapped.unwrapped, "is_rainy") else False)
    estado, _ = env_vis.reset()

    print(f"\n{'─'*50}")
    print(f"  {titulo} – Ejemplo de episodio (máx {n_pasos} pasos)")
    print(f"{'─'*50}")
    print(f"  Decodificación: (fila_taxi, col_taxi, pasajero, destino)")
    print(f"  Acciones: {SIMBOLOS_ACCION}  (↓Sur ↑Norte →Este ←Oeste P=pickup D=dropoff)")
    print()

    recompensa_total = 0
    for paso in range(n_pasos):
        fila, col, pas, dest = env_vis.unwrapped.decode(estado)
        accion = int(policy[estado])
        env_vis.render()     
        siguiente, reward, terminated, truncated, _ = env_vis.step(accion)
        recompensa_total += reward
        print(f"  Paso {paso+1:>2} | estado={estado:>3} ({fila},{col},p={pas},d={dest})"
              f" | acción={SIMBOLOS_ACCION[accion]}({NOMBRES_ACCION[accion]:>7})"
              f" | reward={reward:>4} | r_acum={recompensa_total:>4}")
        estado = siguiente
        if terminated or truncated:
            print(f"  {'─'*46}")
            print(f"  ✓ Episodio terminado en {paso+1} pasos | r_total={recompensa_total}")
            break
    env_vis.close()


def mostrar_qtable_resumen(Q, top_n=10):
    """Muestra los top_n estados con mayor valor máximo en la Q-table."""
    max_q = np.max(Q, axis=1)
    top_estados = np.argsort(max_q)[::-1][:top_n]
    print(f"\n  Top {top_n} estados por valor Q máximo:")
    print(f"  {'Estado':>7} | {'(f,c,p,d)':>12} | {'Mejor acción':>14} | {'Q_max':>7}")
    print(f"  {'─'*7}-+-{'─'*12}-+-{'─'*14}-+-{'─'*7}")

    env_tmp = gym.make("Taxi-v4")
    for s in top_estados:
        fila, col, pas, dest = env_tmp.unwrapped.decode(s)
        mejor_a = int(np.argmax(Q[s]))
        print(f"  {s:>7} | ({fila},{col},{pas},{dest}){' ':>5} | "
              f"{NOMBRES_ACCION[mejor_a]:>14} | {max_q[s]:>7.2f}")
    env_tmp.close()

def graficar_todo(recomp_det, recomp_esto,
                  deltas_vi_det, deltas_vi_esto,
                  vi_det, vi_esto,
                  ql_det, ql_esto,
                  Q_det, Q_esto):

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Taxi-v4 – Análisis completo", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35)

    AZUL   = "#5B9BD5"
    NARANJA= "#E05C25"
    VERDE  = "#70AD47"
    ROJO   = "#C00000"
    GRIS   = "#808080"

    ventana = 300

    # ── Fila 0: DETERMINÍSTICO ─────────────────────────
    # 0-0  Recompensa por episodio
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(recomp_det, alpha=0.2, color=AZUL, linewidth=0.5)
    mm = np.convolve(recomp_det, np.ones(ventana)/ventana, mode="valid")
    ax.plot(range(ventana-1, len(recomp_det)), mm, color=NARANJA, linewidth=1.8,
            label=f"Media móvil {ventana}")
    ax.axhline(0, color=GRIS, linestyle="--", linewidth=0.7)
    ax.set_title("QL Determinístico\nRecompensa por episodio")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 0-1  Convergencia VI determinístico
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(deltas_vi_det, color=VERDE, linewidth=1.5)
    ax.axhline(1e-8, color=ROJO, linestyle="--", linewidth=1, label="θ=1e-8")
    ax.set_title("VI Determinístico\nConvergencia (delta por iter.)")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Delta (escala log)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 0-2  Distribución Q-table det (max Q por estado)
    ax = fig.add_subplot(gs[0, 2])
    max_q_det = np.max(Q_det, axis=1)
    ax.hist(max_q_det[max_q_det != 0], bins=40, color=AZUL, edgecolor="white", alpha=0.85)
    ax.set_title("Det: Distribución max Q(s,·)\n(estados no nulos)")
    ax.set_xlabel("Valor Q máximo"); ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)

    # 0-3  Comparación VI vs QL (determinístico)
    ax = fig.add_subplot(gs[0, 3])
    _plot_comparacion(ax, vi_det, ql_det, VERDE, AZUL, "Determinístico")

    # ── Fila 1: ESTOCÁSTICO ────────────────────────────
    # 1-0  Recompensa por episodio
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(recomp_esto, alpha=0.2, color=ROJO, linewidth=0.5)
    mm2 = np.convolve(recomp_esto, np.ones(ventana)/ventana, mode="valid")
    ax.plot(range(ventana-1, len(recomp_esto)), mm2, color=NARANJA, linewidth=1.8,
            label=f"Media móvil {ventana}")
    ax.axhline(0, color=GRIS, linestyle="--", linewidth=0.7)
    ax.set_title("QL Estocástico (is_rainy)\nRecompensa por episodio")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 1-1  Convergencia VI estocástico
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(deltas_vi_esto, color=ROJO, linewidth=1.5)
    ax.axhline(1e-8, color=ROJO, linestyle="--", linewidth=1, label="θ=1e-8")
    # Comparar curvas en mismo eje
    ax.semilogy(deltas_vi_det, color=VERDE, linewidth=1, alpha=0.6, label="Det (ref)")
    ax.set_title("VI Estocástico\nConvergencia vs Determinístico")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Delta (escala log)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 1-2  Distribución Q-table esto
    ax = fig.add_subplot(gs[1, 2])
    max_q_esto = np.max(Q_esto, axis=1)
    ax.hist(max_q_esto[max_q_esto != 0], bins=40, color=ROJO, edgecolor="white", alpha=0.85)
    ax.set_title("Esto: Distribución max Q(s,·)\n(estados no nulos)")
    ax.set_xlabel("Valor Q máximo"); ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)

    # 1-3  Comparación VI vs QL (estocástico)
    ax = fig.add_subplot(gs[1, 3])
    _plot_comparacion(ax, vi_esto, ql_esto, VERDE, ROJO, "Estocástico")

    plt.savefig("taxi_resultados.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nGráfica guardada → taxi_resultados.png")


def _plot_comparacion(ax, vi_stats, ql_stats, c_vi, c_ql, titulo):
    metodos = ["Value\nIteration", "Q-Learning"]
    medias  = [vi_stats["media_recompensa"], ql_stats["media_recompensa"]]
    stds    = [vi_stats["std_recompensa"],   ql_stats["std_recompensa"]]
    bars = ax.bar(metodos, medias, yerr=stds, capsize=5,
                  color=[c_vi, c_ql], edgecolor="white", width=0.45,
                  error_kw={"elinewidth": 1.2, "ecolor": "#555"})
    for bar, m, s in zip(bars, medias, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.15,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    tabla = (f"{'':12}{'VI':>6}{'QL':>6}\n"
             f"{'R.media':12}{vi_stats['media_recompensa']:>6.1f}{ql_stats['media_recompensa']:>6.1f}\n"
             f"{'Pasos':12}{vi_stats['media_pasos']:>6.1f}{ql_stats['media_pasos']:>6.1f}\n"
             f"{'%Exito':12}{vi_stats['exito_pct']:>5.1f}%{ql_stats['exito_pct']:>5.1f}%")
    ax.text(0.02, 0.02, tabla, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_title(f"{titulo}\nVI vs Q-Learning")
    ax.set_ylabel("Recompensa media")
    ax.grid(True, axis="y", alpha=0.3)


def correr_experimento(is_rainy, n_ql=10_000, n_eval=1000):
    nombre = "ESTOCÁSTICO (is_rainy=True)" if is_rainy else "DETERMINÍSTICO (is_rainy=False)"
    print(f"\n{'='*55}")
    print(f"  Modo: {nombre}")
    print(f"{'='*55}")

    env = gym.make("Taxi-v4", is_rainy=is_rainy)
    raw = env.unwrapped

    # Value Iteration
    print("\n[VI] Ejecutando Value Iteration...")
    V, policy_vi, deltas = value_iteration(raw)

    print(f"\n[VI] Evaluando ({n_eval} episodios)...")
    vi_stats = evaluar_politica(env, policy_vi, n_eval)
    print(f"  R.media={vi_stats['media_recompensa']:.2f}±{vi_stats['std_recompensa']:.2f}"
          f"  pasos={vi_stats['media_pasos']:.1f}  éxito={vi_stats['exito_pct']:.1f}%")

    # Mostrar paso a paso de un episodio con política VI
    mostrar_politica_ejemplo(env, policy_vi,
                             titulo=f"Política VI – {nombre}")

    # Q-Learning
    print(f"\n[QL] Entrenando Q-Learning ({n_ql} episodios)...")
    Q, policy_ql, recompensas = q_learning(env, n_episodes=n_ql)

    print(f"\n[QL] Evaluando ({n_eval} episodios)...")
    ql_stats = evaluar_politica(env, policy_ql, n_eval)
    print(f"  R.media={ql_stats['media_recompensa']:.2f}±{ql_stats['std_recompensa']:.2f}"
          f"  pasos={ql_stats['media_pasos']:.1f}  éxito={ql_stats['exito_pct']:.1f}%")

    mostrar_qtable_resumen(Q)

    env.close()
    return V, policy_vi, deltas, vi_stats, Q, policy_ql, recompensas, ql_stats


if __name__ == "__main__":
    print("=" * 55)
    print("  TAXI-v4  –  Value Iteration + Q-Learning")
    print("  Determinístico  vs  Estocástico (is_rainy)")
    print("=" * 55)

    #  Determinístico 
    (V_det, pol_vi_det, deltas_det,
     vi_det, Q_det, pol_ql_det, recomp_det, ql_det) = correr_experimento(is_rainy=False)

    #  Estocástico
    (V_esto, pol_vi_esto, deltas_esto,
     vi_esto, Q_esto, pol_ql_esto, recomp_esto, ql_esto) = correr_experimento(is_rainy=True)

    #  Comparación final ─
    print(f"\n{'='*60}")
    print("  COMPARACIÓN FINAL")
    print(f"{'='*60}")
    print(f"  {'':22} {'DET-VI':>8} {'DET-QL':>8} {'ESTO-VI':>9} {'ESTO-QL':>8}")
    print(f"  {'Recompensa media':22}"
          f" {vi_det['media_recompensa']:>8.2f} {ql_det['media_recompensa']:>8.2f}"
          f" {vi_esto['media_recompensa']:>9.2f} {ql_esto['media_recompensa']:>8.2f}")
    print(f"  {'Pasos medios':22}"
          f" {vi_det['media_pasos']:>8.1f} {ql_det['media_pasos']:>8.1f}"
          f" {vi_esto['media_pasos']:>9.1f} {ql_esto['media_pasos']:>8.1f}")
    print(f"  {'% Éxito':22}"
          f" {vi_det['exito_pct']:>7.1f}% {ql_det['exito_pct']:>7.1f}%"
          f" {vi_esto['exito_pct']:>8.1f}% {ql_esto['exito_pct']:>7.1f}%")
    print(f"  {'Iter. convergencia VI':22}"
          f" {len(deltas_det):>8}  {'—':>8}"
          f" {len(deltas_esto):>9}  {'—':>8}")

    #  Gráficas 
    print("\n[Graficando...]")
    graficar_todo(recomp_det, recomp_esto,
                  deltas_det, deltas_esto,
                  vi_det, vi_esto,
                  ql_det, ql_esto,
                  Q_det, Q_esto)