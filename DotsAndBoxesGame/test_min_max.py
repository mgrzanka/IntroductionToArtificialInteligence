from State import DostsAndBoxesState
from Game import DotsAndBoxes
import matplotlib.pyplot as plt


def test_generate_child_states():
    state = DostsAndBoxesState(1, 4)
    children = state.generate_children()
    children2 = children[0].generate_children()
    children3 = children2[2].generate_children()
    children4 = children3[2].generate_children()
    children5 = children4[4].generate_children()
    print(children5[0])
    print(children5[0].find_box(0, 0))
    print(children5[0].evaluate_state())
    pass


def test_generate_first_state():
    state = DostsAndBoxesState(1, 3)
    print(state)


# test_generate_child_states()
# test_generate_first_state()


def test_find_box():
    state = DostsAndBoxesState(1, 3)
    state.find_box(0, 0)


def test_game():
    game = DotsAndBoxes()
    game.play()


def test_play_two_bots():
    game = DotsAndBoxes(depth_bot1=5, depth_bot2=5, dimentions=3, first_player=1)
    game.play_two_bots()


winners = []
depth_bot1 = 10
depth_bot2 = 14
dimentions = 3
first_player = 1
game = DotsAndBoxes(depth_bot1=depth_bot1, depth_bot2=depth_bot2, dimentions=dimentions, first_player=first_player)

num_iterations = 8
num_wins_player1 = 0
num_wins_player2 = 0
ties = 0

for _ in range(num_iterations):
    winner = game.play_two_bots()
    if winner == 1:
        num_wins_player1 += 1
    elif winner == -1:
        num_wins_player2 += 1
    else:
        ties += 1


# Procent zwycięstw dla każdej strategii
percent_wins_player1 = (num_wins_player1 / num_iterations) * 100
percent_wins_player2 = (num_wins_player2 / num_iterations) * 100
percent_ties = (ties/num_iterations) * 100

# Tworzenie wykresu kołowego
labels = ['Max', 'Min', 'Ties']
sizes = [percent_wins_player1, percent_wins_player2, percent_ties]
colors = ['lightcoral', 'lightskyblue', 'red']
explode = (0, 0, 0)  # Explode pierwszej części
print(percent_wins_player1)
print(percent_wins_player2)
print(percent_ties)

plt.axis('equal')  # Ustawienie wykresu kołowego jako koła
plt.title(f"Percentage of Wins for depth_bot1={depth_bot1}, depth_bot2={depth_bot2}, dimensions={dimentions}, first_player={'Max' if first_player else 'Min'}")
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()
