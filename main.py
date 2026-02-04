import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm

class MountainCarBase:
    def __init__(self, n_position_bins=20, n_velocity_bins=20, gamma=0.99, theta=1e-6):
        self.env = gym.make('MountainCar-v0')
        self.gamma = gamma
        self.theta = theta
        
        self.n_position_bins = n_position_bins
        self.n_velocity_bins = n_velocity_bins
        self.n_states = n_position_bins * n_velocity_bins
        self.n_actions = self.env.action_space.n
        
        self.position_bins = np.linspace(-1.2, 0.6, n_position_bins + 1)
        self.velocity_bins = np.linspace(-0.07, 0.07, n_velocity_bins + 1)
        
        self.V = np.zeros(self.n_states)
        self.policy = None
        
        self.iterations = 0
        self.value_history = []
        
    def discretize_state(self, state):
        position, velocity = state
        pos_bin = np.digitize(position, self.position_bins) - 1
        pos_bin = np.clip(pos_bin, 0, self.n_position_bins - 1)
        vel_bin = np.digitize(velocity, self.velocity_bins) - 1
        vel_bin = np.clip(vel_bin, 0, self.n_velocity_bins - 1)
        return pos_bin * self.n_velocity_bins + vel_bin
    
    def sample_transitions(self, n_samples=100):
        transitions = defaultdict(list)
        print("Збір семплів переходів...")
        
        for episode in tqdm(range(n_samples)):
            state, _ = self.env.reset()
            
            if episode % 3 == 0:
                random_pos = np.random.uniform(-1.2, 0.5)
                random_vel = np.random.uniform(-0.07, 0.07)
                self.env.unwrapped.state = np.array([random_pos, random_vel])
                state = self.env.unwrapped.state
            
            done = False
            steps = 0
            max_steps = 200
            
            while not done and steps < max_steps:
                current_s = self.discretize_state(state)
                
                for a in range(self.n_actions):
                    saved_state = state.copy()
                    self.env.unwrapped.state = saved_state
                    next_state, reward, terminated, truncated, _ = self.env.step(a)
                    done_flag = terminated or truncated
                    s_next = self.discretize_state(next_state)
                    transitions[(current_s, a)].append((s_next, reward, done_flag))
                    self.env.unwrapped.state = saved_state
                
                if np.random.random() < 0.3:
                    action = self.env.action_space.sample()
                else:
                    action = steps % 3
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                steps += 1
        
        return transitions
    
    def get_transition_prob(self, transitions):
        P = defaultdict(float)
        R = defaultdict(float)
        counts = defaultdict(int)
        
        for (s, a), samples in transitions.items():
            for s_next, reward, done in samples:
                P[(s, a, s_next)] += 1
                R[(s, a, s_next)] += reward
                counts[(s, a)] += 1
        
        for key in P:
            s, a, s_next = key
            P[key] /= counts[(s, a)]
            R[key] /= (P[key] * counts[(s, a)])
        
        return P, R
    
    def test_policy(self, n_episodes=10, render=False):
        if self.policy is None:
            print("Спочатку навчіть модель!")
            return None, None
        
        print(f"\nТестування політики на {n_episodes} епізодах...")
        
        if render:
            test_env = gym.make('MountainCar-v0', render_mode='human')
        else:
            test_env = gym.make('MountainCar-v0')
        
        total_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, _ = test_env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 200:
                s = self.discretize_state(state)
                action = self.policy[s]
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            if not render or n_episodes <= 10:
                print(f"Епізод {episode + 1}: винагорода = {total_reward}, кроків = {steps}")
        
        test_env.close()
        
        print(f"\nСередня винагорода: {np.mean(total_rewards):.2f}")
        print(f"Середня довжина епізоду: {np.mean(episode_lengths):.2f}")
        
        return total_rewards, episode_lengths

class MountainCarValueIteration(MountainCarBase):
    def __init__(self, n_position_bins=20, n_velocity_bins=20, gamma=0.99, theta=1e-6):
        super().__init__(n_position_bins, n_velocity_bins, gamma, theta)
        self.delta_history = []
    
    def value_iteration_step(self, P, R):
        delta = 0
        new_V = self.V.copy()
        
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            has_transitions = False
            
            for a in range(self.n_actions):
                q_value = 0
                for s_next in range(self.n_states):
                    if (s, a, s_next) in P:
                        has_transitions = True
                        prob = P[(s, a, s_next)]
                        reward = R[(s, a, s_next)]
                        q_value += prob * (reward + self.gamma * self.V[s_next])
                action_values[a] = q_value
            
            if has_transitions:
                new_V[s] = np.max(action_values)
                delta = max(delta, abs(self.V[s] - new_V[s]))
        
        self.V = new_V
        return delta
    
    def extract_policy(self, P, R):
        policy = np.zeros(self.n_states, dtype=int)
        
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    if (s, a, s_next) in P:
                        prob = P[(s, a, s_next)]
                        reward = R[(s, a, s_next)]
                        action_values[a] += prob * (reward + self.gamma * self.V[s_next])
            
            policy[s] = np.argmax(action_values)
        
        return policy
    
    def train(self, max_iterations=1000, n_samples=1000):
        print("Початок Value Iteration для Mountain Car")
        print(f"Дискретизація: {self.n_position_bins}x{self.n_velocity_bins} = {self.n_states} станів")
        
        transitions = self.sample_transitions(n_samples)
        P, R = self.get_transition_prob(transitions)
        
        print(f"\nЗібрано {len(transitions)} унікальних пар (стан, дія)")
        print(f"Початок ітерацій Value Iteration...\n")
        
        for iteration in range(max_iterations):
            delta = self.value_iteration_step(P, R)
            
            self.value_history.append(np.mean(self.V))
            self.delta_history.append(delta)
            
            if (iteration + 1) % 10 == 0:
                print(f"Ітерація {iteration + 1}: delta = {delta:.6f}"
                      )
            
            self.iterations = iteration + 1
            
            if delta < self.theta:
                print(f"\nValue Iteration збігся після {iteration + 1} ітерацій!")
                print(f"Фінальна delta: {delta:.8f}")
                break
        
        self.policy = self.extract_policy(P, R)
        print(f"\nПолітику витягнуто!")
        
        return self.policy, self.V

class MountainCarPolicyIteration(MountainCarBase):
    def __init__(self, n_position_bins=20, n_velocity_bins=20, gamma=0.99, theta=1e-6):
        super().__init__(n_position_bins, n_velocity_bins, gamma, theta)
        self.policy = np.zeros(self.n_states, dtype=int)
    
    def policy_evaluation(self, P, R):
        while True:
            delta = 0
            new_V = self.V.copy()
            
            for s in range(self.n_states):
                a = self.policy[s]
                v = 0
                has_transitions = False
                
                for s_next in range(self.n_states):
                    if (s, a, s_next) in P:
                        has_transitions = True
                        prob = P[(s, a, s_next)]
                        reward = R[(s, a, s_next)]
                        v += prob * (reward + self.gamma * self.V[s_next])
                
                if has_transitions:
                    new_V[s] = v
                    delta = max(delta, abs(self.V[s] - new_V[s]))
            
            self.V = new_V
            
            if delta < self.theta:
                break
    
    def policy_improvement(self, P, R):
        policy_stable = True
        
        for s in range(self.n_states):
            old_action = self.policy[s]
            
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    if (s, a, s_next) in P:
                        prob = P[(s, a, s_next)]
                        reward = R[(s, a, s_next)]
                        action_values[a] += prob * (reward + self.gamma * self.V[s_next])
            
            best_action = np.argmax(action_values)
            self.policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def train(self, max_iterations=100, n_samples=1000):
        print("Початок Policy Iteration для Mountain Car")
        print(f"Дискретизація: {self.n_position_bins}x{self.n_velocity_bins} = {self.n_states} станів")
        
        transitions = self.sample_transitions(n_samples)
        P, R = self.get_transition_prob(transitions)
        
        print(f"\nЗібрано {len(transitions)} унікальних пар (стан, дія)")
        print(f"Початок ітерацій Policy Iteration...\n")
        
        for iteration in range(max_iterations):
            self.policy_evaluation(P, R)
            self.value_history.append(np.mean(self.V))
            policy_stable = self.policy_improvement(P, R)
            
            print(f"Ітерація {iteration + 1}")
            
            self.iterations = iteration + 1
            
            if policy_stable:
                print(f"\nПолітика стабілізувалася після {iteration + 1} ітерацій!")
                break
        
        return self.policy, self.V

def run_value_iteration():
    print("\n" + "=" * 70)
    print("VALUE ITERATION")
    print("=" * 70)
    
    agent = MountainCarValueIteration(
        n_position_bins=40,
        n_velocity_bins=40,
        gamma=0.95,
        theta=1e-4
    )
    agent.train(max_iterations=2000, n_samples=5000)
    
    agent.test_policy(n_episodes=20, render=False)
    
    print("\n" + "=" * 70)
    print("ВІЗУАЛІЗАЦІЯ (3 епізоди)")
    print("=" * 70)
    agent.test_policy(n_episodes=3, render=True)
    
    return agent


def run_policy_iteration():
    print("\n" + "=" * 70)
    print("POLICY ITERATION")
    print("=" * 70)
    
    agent = MountainCarPolicyIteration(
        n_position_bins=40,
        n_velocity_bins=40,
        gamma=0.95,
        theta=1e-4
    )
    agent.train(max_iterations=50, n_samples=5000)
    
    agent.test_policy(n_episodes=20, render=False)
    
    print("\n" + "=" * 70)
    print("ВІЗУАЛІЗАЦІЯ (3 епізоди)")
    print("=" * 70)
    agent.test_policy(n_episodes=3, render=True)
    
    return agent

def interactive_menu():
    agent = None
    
    while True:
        print("\n1. Value Iteration")
        print("2. Policy Iteration")
        print("3. Демонстрація останньої моделі")
        print("0. Вихід")
        
        choice = input("\nВаш вибір: ").strip()
        
        if choice == "1":
            agent = run_value_iteration()
        elif choice == "2":
            agent = run_policy_iteration()
        elif choice == "3":
            if agent is None:
                print("\n✗ Спочатку навчіть модель!")
            else:
                n = int(input("\nСкільки епізодів показати? (1-10): ") or "3")
                agent.test_policy(n_episodes=min(max(1, n), 10), render=True)
        elif choice == "0":
            break
        else:
            print("\nНевірний вибір!")

if __name__ == "__main__":
    interactive_menu()
