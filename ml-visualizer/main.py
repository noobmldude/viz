import argparse
from linear_regression import run_linear_regression
from kmeans import run_kmeans
from gradient_descent import run_gradient_descent
from digit_classifier import run_digit_classifier

def main():
    parser = argparse.ArgumentParser(description="ML Algorithm Visualizer")
    parser.add_argument('algorithm', choices=['linear_regression', 'kmeans', 'gradient_descent', 'digit_classifier'],
                        help='The algorithm to visualize.')
    args = parser.parse_args()

    if args.algorithm == 'linear_regression':
        run_linear_regression()
    elif args.algorithm == 'kmeans':
        run_kmeans()
    elif args.algorithm == 'gradient_descent':
        run_gradient_descent()
    elif args.algorithm == 'digit_classifier':
        run_digit_classifier()


if __name__ == '__main__':
    main()
