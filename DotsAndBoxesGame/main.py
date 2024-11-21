import argparse
import sys

from Game import DotsAndBoxes


def create_argments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=4, help="dimentions of the game")
    parser.add_argument("-b1", type=int, default=4, help="depth of bot 1")
    parser.add_argument("-b2", type=int, default=None, help="depth of bot 2")
    parser.add_argument("-pm", type=int, default=1, help="mode of the player (1 for max, 0 for min)")
    parser.add_argument("-f", type=int, default=1, help="who moves first - min or max")
    return parser.parse_args(arguments[1:])


def manage_arguments(arguments):
    args = create_argments(arguments)
    if args.d:
        dimentions = args.d
    if args.b1:
        depth_bot1 = args.b1
    if args.b2:
        depth_bot2 = args.b2
    else:
        depth_bot2 = None
    if args.pm:
        player_mode = args.pm
    if args.f:
        first_player = args.f
    return dimentions, depth_bot1, depth_bot2, player_mode, first_player


def main(arguments):
    game = DotsAndBoxes(*manage_arguments(arguments))
    depth_bot2 = manage_arguments(arguments)[2]
    if depth_bot2 is None:
        game.play()
    else:
        game.play_two_bots()


if __name__ == '__main__':
    main(sys.argv)
