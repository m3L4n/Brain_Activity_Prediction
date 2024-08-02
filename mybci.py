from preprocessing_data import preprocessing_data
from plot_data import plot_data
from predict_data import predict_data
from test_all_data import test_all_data
from train_data import train_data
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=" train or predict or all train and predict"
    )
    parser.add_argument(
        "--action",
        choices=["plot", "train", "predict"],
        action="store",
        help="choose the type of cumputing ",
    )
    parser.add_argument(
        "--subject",
        type=int,
        action="store",
        choices=range(1, 110),
        help="subject number",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        action="store",
        choices=range(3, 15),
        help="subject number",
    )
    function_action = None
    args = parser.parse_args()
    action = args.action
    subject = args.subject
    tasks = args.tasks
    print(args, action, subject, tasks)
    if action is None:
        test_all_data()
    else:
        match action:
            case "plot":
                plot_data(subject, tasks)
                return
            case "train":
                function_action = train_data
            case "predict":

                function_action = predict_data
            case _:
                parser.print_help()
                return
        X, y = preprocessing_data(tasks, subject)
        return function_action(X, y)


if __name__ == "__main__":
    main()
