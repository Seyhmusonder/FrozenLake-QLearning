"""
Pekiştirmeli Öğrenme Ödevi
Problem / Ortam : FrozenLake-v1 (4x4, stochastic=True)
Yöntem          : Q-Learning (Tablo Tabanlı, Model-Free, Off-Policy)

Açıklama:
FrozenLake, bir ajanın buzlu bir gölün üzerinde başlangıç noktasından
hedef noktasına, deliklere (H) düşmeden gitmeye çalıştığı klasik bir
RL ortamıdır. Durum uzayı 16 (4x4 grid), aksiyon uzayı 4'tür
(Sol, Aşağı, Sağ, Yukarı). Buz kaygan olduğu için (is_slippery=True)
seçilen aksiyon her zaman istenen yöne gitmeyebilir; bu da ortamı
stokastik (rastgele) hale getirir.

Bu script:
1. Ortamı oluşturur
2. Q-Learning algoritması ile ajanı eğitir
3. Eğitim sürecindeki ödül grafiğini çizer ve kaydeder
4. Öğrenilen Q-tablosunu CSV olarak kaydeder (rapora eklenecek "veri seti")
5. Eğitilmiş ajanı test ederek başarı oranını hesaplar
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import csv

# ---------------------------------------------------------
# 1) Ortam ve Hiperparametreler
# ---------------------------------------------------------
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

n_states = env.observation_space.n   # 16
n_actions = env.action_space.n       # 4

# Hiperparametreler
alpha = 0.8          # öğrenme oranı (learning rate)
gamma = 0.95         # gelecek ödüllerin indirim faktörü (discount factor)
epsilon = 1.0        # başlangıç keşif (exploration) oranı
epsilon_min = 0.01
epsilon_decay = 0.9995

n_episodes = 10000
max_steps = 100

# Q-tablosu: satırlar=durumlar, sütunlar=aksiyonlar -> 0 ile başlat
Q = np.zeros((n_states, n_actions))

rewards_per_episode = []

# ---------------------------------------------------------
# 2) Q-Learning Eğitim Döngüsü
# ---------------------------------------------------------
for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy aksiyon seçimi
        if np.random.rand() < epsilon:
            action = env.action_space.sample()          # keşif (explore)
        else:
            action = np.argmax(Q[state, :])              # sömürü (exploit)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning güncelleme formülü:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max(Q(s', :)) - Q(s,a)]
        best_next = np.max(Q[next_state, :])
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * best_next - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    # Epsilon'u kademeli olarak azalt (önce çok keşfet, sonra öğrendiğini kullan)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

env.close()

# ---------------------------------------------------------
# 3) Eğitim Grafiği (her 100 bölümün ortalama ödülü)
# ---------------------------------------------------------
window = 100
avg_rewards = [
    np.mean(rewards_per_episode[max(0, i - window):i + 1])
    for i in range(len(rewards_per_episode))
]

plt.figure(figsize=(10, 5))
plt.plot(avg_rewards)
plt.xlabel("Bölüm (Episode)")
plt.ylabel("Son 100 bölümün ortalama ödülü")
plt.title("Q-Learning Eğitim Performansı - FrozenLake 4x4")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_rewards.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 4) Q-Tablosunu CSV olarak kaydet (rapora eklenecek veri seti)
# ---------------------------------------------------------
action_names = ["Sol", "Asagi", "Sag", "Yukari"]
with open("q_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["State"] + action_names)
    for s in range(n_states):
        writer.writerow([s] + list(np.round(Q[s, :], 4)))

# ---------------------------------------------------------
# 5) Eğitilmiş Ajanı Test Et (başarı oranı)
# ---------------------------------------------------------
test_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
n_test_episodes = 1000
successes = 0

for episode in range(n_test_episodes):
    state, _ = test_env.reset()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])  # tamamen öğrenilen politikayı kullan
        state, reward, terminated, truncated, _ = test_env.step(action)
        if terminated or truncated:
            if reward == 1:
                successes += 1
            break

test_env.close()

success_rate = successes / n_test_episodes * 100

print("=" * 50)
print("EGITIM TAMAMLANDI")
print(f"Toplam bolum sayisi          : {n_episodes}")
print(f"Test bolum sayisi            : {n_test_episodes}")
print(f"Test basari orani (Q-Learning): %{success_rate:.2f}")
print(f"Egitim sonu epsilon          : {epsilon:.4f}")
print("Q-tablosu 'q_table.csv' olarak kaydedildi.")
print("Egitim grafigi 'training_rewards.png' olarak kaydedildi.")
print("=" * 50)

# Q-tablosunu da ekrana yazdir
np.set_printoptions(precision=3, suppress=True)
print("\nOgrenilen Q-Tablosu (16 durum x 4 aksiyon):")
print("Aksiyonlar: [Sol, Asagi, Sag, Yukari]")
print(Q)
