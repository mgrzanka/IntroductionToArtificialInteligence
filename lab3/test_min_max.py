from State import DostsAndBoxesState
from Game import DotsAndBoxes


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
    game = DotsAndBoxes(depth_bot1=1, depth_bot2=10)
    game.play_two_bots()


test_game()
# test_play_two_bots()
