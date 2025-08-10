import argparse
from linear_regression import run_linear_regression
from kmeans import run_kmeans
from gradient_descent import run_gradient_descent

def main():
    parser = argparse.ArgumentParser(description="ML Algorithm Visualizer")
    parser.add_argument('algorithm', choices=['linear_regression', 'kmeans', 'gradient_descent'],
                        help='The algorithm to visualize.')
    args = parser.parse_args()

    if args.algorithm == 'linear_regression':
        run_linear_regression()
    elif args.algorithm == 'kmeans':
        run_kmeans()
    elif args.algorithm == 'gradient_descent':
        run_gradient_descent()


if __name__ == '__main__':
    main()
