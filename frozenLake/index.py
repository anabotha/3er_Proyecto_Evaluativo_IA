"""
Frozen Lake - Value Iteration
==============================
Punto 1, Parte A: Método basado en modelo

El agente aprende una política óptima usando Value Iteration,
sin necesidad de explorar el entorno con prueba y error.
Usa directamente la dinámica del entorno: env.P[s][a]
"""

import gymnasium as gym
import numpy as np



def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states  = env.observation_space.n   # 16 estados (grilla 4x4)
    n_actions = env.action_space.n        # arriba,abajo,der,izq

    V = np.zeros(n_states)
    
    iteration = 0
    while True:
        delta = 0  # registra el cambio máximo en esta iteración

        # PASO 2: Recorrer todos los estados
        for s in range(n_states):

            
            v_actual = V[s]

            valores_por_accion = []

            for a in range(n_actions):

              
                q_sa = 0  # valor de tomar la acción 'a' en el estado 's'

                for (prob, next_state, reward, done) in env.P[s][a]:
                    # Ecuación de Bellman:
                    # Q(s,a) = Σ prob * (reward + gamma * V[s'])
                    # Si 'done' es True (agujero o meta), V[s'] = 0
                    q_sa += prob * (reward + gamma * V[next_state])

                valores_por_accion.append(q_sa)

         #se guarad el valor de la accion con mejor puntaje
            V[s] = max(valores_por_accion)


            delta = max(delta, abs(v_actual - V[s]))

        iteration += 1

   
        if delta < theta:
            print(f"Convergió en {iteration} iteraciones (delta={delta:.2e})")
            break

    # Para cada estado, elegir la acción con mayor Q(s,a)
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        valores_por_accion = []

        for a in range(n_actions):
            q_sa = 0
            for (prob, next_state, reward, done) in env.P[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            valores_por_accion.append(q_sa)

        # La política elige la acción que maximiza Q(s,a)
        policy[s] = np.argmax(valores_por_accion)

    return V, policy


def evaluar_politica(env, policy, n_episodios=1000):
    """
    Ejecuta n_episodios usando la política dada y
    devuelve el porcentaje de éxito (llegar a la meta).
    """
    exitos = 0

    for _ in range(n_episodios):
        estado, _ = env.reset()
        done = False

        while not done:
            accion = policy[estado]                    # seguir la política
            estado, reward, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated

        if reward == 1.0:   # recompensa 1.0 = llegó a la meta
            exitos += 1

    tasa_exito = exitos / n_episodios * 100
    return tasa_exito


def mostrar_politica(policy, V):
    """Imprime la política y los valores en forma de grilla 4x4."""
    # Mapeo de acción a símbolo: 0=←  1=↓  2=→  3=↑
    simbolos = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    # Posiciones de agujeros y meta en Frozen Lake estándar
    agujeros = {5, 7, 11, 12}
    meta     = {15}

    print("\n── Política óptima (grilla 4x4) ──")
    print("  Leyenda: ← ↓ → ↑  |  H=agujero  G=meta\n")

    for s in range(16):
        if s in agujeros:
            simbolo = " H"
        elif s in meta:
            simbolo = " G"
        else:
            simbolo = f" {simbolos[policy[s]]}"

        print(simbolo, end="")

        if (s + 1) % 4 == 0:
            print()  # salto de línea cada 4 estados

    print("\n── Función de valor V(s) ──")
    for s in range(16):
        print(f"{V[s]:.3f}", end="  ")
        if (s + 1) % 4 == 0:
            print()

def q_learning(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
    """
    Implementa el algoritmo Q-Learning.
    """
    # 1. Inicializar la tabla Q con ceros
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    # Lista para guardar las recompensas y luego graficarlas
    rewards = []
    
    for episode in range(episodes):
        # Reiniciar el entorno
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 2. Aplicar estrategia epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Exploración (acción aleatoria) 
            else:
                action = np.argmax(Q[state, :])    # Explotación (mejor acción conocida) 
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. Actualizar Q mediante la ecuación de aprendizaje (Ecuación de Bellman) 
            # Si el episodio terminó, el valor futuro esperado es 0
            best_next_value = np.max(Q[next_state, :]) if not done else 0.0
            
            # Ecuación de actualización
            td_target = reward + gamma * best_next_value
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
            
            # Actualizar el estado actual y sumar recompensa
            state = next_state
            total_reward += reward
            
        # Decaimiento de epsilon al final de cada episodio (con un límite mínimo) 
        epsilon = max(0.01, epsilon * epsilon_decay)
        
        # Guardar la recompensa del episodio
        rewards.append(total_reward)
        
    # Obtener la política derivada de Q (la mejor acción para cada estado)
    policy = np.argmax(Q, axis=1)
    
    # Retorna Q, policy, y rewards
    return Q, policy, rewards

def correr_experimento(modo_slippery, n_episodios=1000):
   
    nombre = "ESTOCÁSTICO (is_slippery=True)" if modo_slippery else "DETERMINÍSTICO (is_slippery=False)"
    print(f"\n{'='*55}")
    print(f"  Modo: {nombre}")
    print(f"{'='*55}")

    # entorno
    # env_wrapped: para reset/step en episodios de prueba
    # env_unwrapped: para acceder a env.P (dinámica del modelo)
    env_wrapped   = gym.make("FrozenLake-v1", is_slippery=modo_slippery)
    env_unwrapped = env_wrapped.unwrapped


    V, policy = value_iteration(env_unwrapped, gamma=0.99, theta=1e-8)

    # Mostrar política y valores
    mostrar_politica(policy, V)

    tasa = evaluar_politica(env_wrapped, policy, n_episodios=n_episodios)
    print(f"\n Tasa de éxito ({n_episodios} episodios): {tasa:.1f}%")

    env_wrapped.close()
    return V, policy, tasa


if __name__ == "__main__":

    # Modo determinístico
    V_det, policy_det, tasa_det = correr_experimento(modo_slippery=False)

    # Modo estocástico
    V_esto, policy_esto, tasa_esto = correr_experimento(modo_slippery=True)

    # Comparación final
    print(f"\n{'='*55}")
    print("  COMPARACIÓN FINAL")
    print(f"{'='*55}")
    print(f"  Determinístico : {tasa_det:.1f}% de éxito")
    print(f"  Estocástico    : {tasa_esto:.1f}% de éxito")
    print(f"\n  Diferencia     : {tasa_det - tasa_esto:.1f} puntos porcentuales")

    '''
    Q-Learning
    q_learning(env,
        episodes=10000, 
        alpha=0.1, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.999)
    '''
    # Modo Determinístico 
    print("Entrenando en modo determinístico...")
    env_deterministico = gym.make("FrozenLake-v1", is_slippery=False)

    # Le pasamos el juego a la función
    Q_det, policy_det, rewards_det = q_learning(env_deterministico)


    # Modo Estocástico
    print("Entrenando en modo estocástico...")
    env_estocastico = gym.make("FrozenLake-v1", is_slippery=True)

    # Le pasamos el nuevo juego a la función 
    Q_est, policy_est, rewards_est = q_learning(env_estocastico)