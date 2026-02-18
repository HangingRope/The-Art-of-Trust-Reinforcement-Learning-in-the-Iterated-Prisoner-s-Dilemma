import pygame
import random
import sys
from dataclasses import dataclass
from statistics import mean

# -------------------------
# Payoffs & utils
# -------------------------
def payoff(player, agent, T=5, R=3, P=1, S=0):
    if player == "C" and agent == "C":
        return R, R
    if player == "C" and agent == "D":
        return S, T
    if player == "D" and agent == "C":
        return T, S
    return P, P

ACTIONS = ["C", "D"]

def next_state(prev_player, prev_agent):
    if prev_player is None or prev_agent is None:
        return "START"
    return prev_player + prev_agent

@dataclass
class RoundResult:
    round_no: int
    player: str
    agent: str
    player_reward: int
    agent_reward: int

# -------------------------
# Q-learning Agent
# -------------------------
class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=0.25, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def _get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        qs = [self._get_q(state, a) for a in ACTIONS]
        max_q = max(qs)
        best = [a for a, q in zip(ACTIONS, qs) if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state):
        old_q = self._get_q(state, action)
        future_q = max(self._get_q(next_state, a) for a in ACTIONS)
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.Q[(state, action)] = new_q

    def decay(self):
        self.epsilon *= self.epsilon_decay

    def pretrain_self_play(self, episodes=2000, rounds_per_episode=5):
        for _ in range(episodes):
            prev_p, prev_a = None, None
            for _ in range(rounds_per_episode):
                s = next_state(prev_p, prev_a)
                a1 = self.choose_action(s)
                a2 = self.choose_action(s)
                r1, r2 = payoff(a1, a2)
                s2 = next_state(a1, a2)
                self.update(s, a1, r1, s2)
                self.update(s, a2, r2, s2)
                prev_p, prev_a = a1, a2
        self.epsilon = 0.03

# -------------------------
# Pygame UI
# -------------------------
pygame.init()
WIDTH, HEIGHT = 950, 580   # smaller window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Prisoner's Dilemma - Smart RL Agent")
font = pygame.font.SysFont("Arial", 20)
small = pygame.font.SysFont("Arial", 14)
mini = pygame.font.SysFont("Arial", 12)
clock = pygame.time.Clock()

# Colors & theme
BG = (24, 26, 30)
PANEL = (34, 38, 45)
CARD_BG = (250, 250, 250)
TEXT = (28, 30, 34)
LINE = (80, 84, 92)
PLAYER_COOP = (70, 130, 180)
PLAYER_DEF = (255, 140, 0)
AGENT_COOP = (34, 139, 34)
AGENT_DEF = (220, 20, 60)
ROUND_COL = (60, 60, 70)

def draw_text(surf, txt, x, y, f, color=TEXT, center=False):
    img = f.render(txt, True, color)
    r = img.get_rect()
    if center:
        r.center = (x, y)
    else:
        r.topleft = (x, y)
    surf.blit(img, r)

def draw_round_tree(surface, base_x, base_y, rr: RoundResult):
    root_x, root_y = base_x, base_y
    horiz = 70
    leaf_y = root_y + 28

    pygame.draw.line(surface, LINE, (root_x, root_y), (root_x - horiz, leaf_y), 2)
    pygame.draw.line(surface, LINE, (root_x, root_y), (root_x + horiz, leaf_y), 2)

    pygame.draw.circle(surface, ROUND_COL, (root_x, root_y), 16)
    draw_text(surface, f"R{rr.round_no}", root_x, root_y-9, small, color=(240,240,240), center=True)

    px, py = root_x - horiz, leaf_y
    pcol = PLAYER_COOP if rr.player == "C" else PLAYER_DEF
    pygame.draw.circle(surface, pcol, (px, py), 14)
    draw_text(surface, rr.player, px, py-8, small, color=(255,255,255), center=True)

    ax, ay = root_x + horiz, leaf_y
    acol = AGENT_COOP if rr.agent == "C" else AGENT_DEF
    pygame.draw.circle(surface, acol, (ax, ay), 14)
    draw_text(surface, rr.agent, ax, ay-8, small, color=(255,255,255), center=True)

    draw_text(surface, f"{rr.player_reward} | {rr.agent_reward}", root_x, leaf_y+22, small, color=LINE, center=True)

def draw_graph(surface, rect, history, max_rounds):
    x, y, w, h = rect
    pygame.draw.rect(surface, CARD_BG, rect, border_radius=8)
    pygame.draw.rect(surface, LINE, rect, 2, border_radius=8)
    draw_text(surface, "Cooperation (you=green, agent=blue)", x+10, y+6, mini, color=TEXT)

    if len(history) < 2:
        return

    margin = 30
    usable_w = w - 2*margin
    n = min(len(history), max_rounds)
    xs = [x + margin + usable_w * (i / (max_rounds-1 if max_rounds>1 else 1)) for i in range(max_rounds)]
    xs_recent = xs[-n:]

    recent = history[-n:]
    p_vals = [1.0 if r.player == 'C' else 0.0 for r in recent]
    a_vals = [1.0 if r.agent == 'C' else 0.0 for r in recent]

    def y_from_val(v):
        return y + h - margin - int(v * (h - 2*margin))

    for i in range(len(xs_recent)-1):
        pygame.draw.line(surface, (34,139,34), (xs_recent[i], y_from_val(p_vals[i])),
                         (xs_recent[i+1], y_from_val(p_vals[i+1])), 2)
        pygame.draw.line(surface, (70,130,180), (xs_recent[i], y_from_val(a_vals[i])),
                         (xs_recent[i+1], y_from_val(a_vals[i+1])), 2)

    window = 6
    if len(p_vals) >= window:
        p_avg = [mean(p_vals[i:i+window]) for i in range(len(p_vals)-window+1)]
        a_avg = [mean(a_vals[i:i+window]) for i in range(len(a_vals)-window+1)]
        xs_avg = xs_recent[window-1:]
        for i in range(len(xs_avg)-1):
            pygame.draw.line(surface, (144,238,144), (xs_avg[i], y_from_val(p_avg[i])),
                             (xs_avg[i+1], y_from_val(p_avg[i+1])), 3)
            pygame.draw.line(surface, (173,216,230), (xs_avg[i], y_from_val(a_avg[i])),
                             (xs_avg[i+1], y_from_val(a_avg[i+1])), 3)

def percent_coop(history, who='player'):
    if not history:
        return 0.0
    if who == 'player':
        return sum(1 for r in history if r.player == 'C')/len(history) * 100
    return sum(1 for r in history if r.agent == 'C')/len(history) * 100

# -------------------------
# Main gameplay function
# -------------------------
def play_game(total_rounds=30, pretrain_episodes=2000, episode_length=6):
    left_w = 260
    graph_h = 120
    tree_area_x = left_w + 20
    tree_area_w = WIDTH - tree_area_x - 20

    agent = QLearningAgent(alpha=0.15, gamma=0.95, epsilon=0.25, epsilon_decay=0.995)
    agent.pretrain_self_play(episodes=pretrain_episodes, rounds_per_episode=episode_length)

    prev_player = None
    prev_agent = None
    round_no = 1
    player_score = 0
    agent_score = 0
    history = []

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running = False
                elif ev.key == pygame.K_r:
                    agent = QLearningAgent(alpha=0.15, gamma=0.95, epsilon=0.25, epsilon_decay=0.995)
                    agent.pretrain_self_play(episodes=pretrain_episodes, rounds_per_episode=episode_length)
                    prev_player = prev_agent = None
                    round_no, player_score, agent_score = 1, 0, 0
                    history.clear()
                elif ev.key in (pygame.K_c, pygame.K_d) and round_no <= total_rounds:
                    human = 'C' if ev.key == pygame.K_c else 'D'
                    state = next_state(prev_player, prev_agent)
                    a_action = agent.choose_action(state)
                    pr, ar = payoff(human, a_action)
                    player_score += pr
                    agent_score += ar
                    next_s = next_state(human, a_action)
                    agent.update(state, a_action, ar, next_s)
                    agent.decay()
                    history.append(RoundResult(round_no, human, a_action, pr, ar))
                    prev_player, prev_agent = human, a_action
                    round_no += 1

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, left_w, HEIGHT))

        draw_text(screen, "Prisoner's Dilemma", 16, 14, font, color=(220,220,220))
        draw_text(screen, f"Round: {min(round_no, total_rounds)}/{total_rounds}", 16, 48, small, color=(200,200,200))
        draw_text(screen, f"You score: {player_score}", 16, 76, small, color=(70,130,180))
        draw_text(screen, f"Agent score: {agent_score}", 16, 98, small, color=(220,20,60))
        draw_text(screen, f"Diff: {player_score - agent_score}", 16, 120, small, color=(220,220,220))

        draw_text(screen, "Agent params:", 16, 160, small, color=(220,220,220))
        draw_text(screen, f"alpha: {agent.alpha:.2f}", 16, 184, mini, color=(200,200,200))
        draw_text(screen, f"gamma: {agent.gamma:.2f}", 16, 200, mini, color=(200,200,200))
        draw_text(screen, f"epsilon: {agent.epsilon:.3f}", 16, 216, mini, color=(200,200,200))
        draw_text(screen, f"Pretrained: yes", 16, 240, mini, color=(180,255,180))

        draw_text(screen, "Cooperation:", 16, 280, small, color=(220,220,220))
        draw_text(screen, f"You: {percent_coop(history,'player'):.1f} %", 16, 304, mini, color=(70,130,180))
        draw_text(screen, f"Agent: {percent_coop(history,'agent'):.1f} %", 16, 320, mini, color=(34,139,34))

        draw_text(screen, "Controls: C/D = move, R = reset, ESC = quit", 12, HEIGHT-28, mini, color=(180,180,180))

        graph_rect = (tree_area_x, 12, WIDTH - tree_area_x - 12, graph_h)
        draw_graph(screen, graph_rect, history, total_rounds)

        # --- Compressed tree that always fits ---
        base_y = graph_rect[1] + graph_h + 16
        max_tree_h = HEIGHT - base_y - 40
        if history:
            visible_rounds = min(len(history), 8)  # last 8 rounds
            row_gap = max_tree_h // visible_rounds
            tree_center_x = tree_area_w // 2 + tree_area_x
            for i, rr in enumerate(history[-visible_rounds:]):
                y = base_y + i * row_gap
                draw_round_tree(screen, tree_center_x, y, rr)

        if round_no > total_rounds:
            draw_text(screen, "Game Over - press R to restart", tree_area_x + 20, HEIGHT - 30, small, color=(255,160,160))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_game(total_rounds=30, pretrain_episodes=2000, episode_length=6)

