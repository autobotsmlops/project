import sys
import pandas as pd


def check_metrics():
    # get metrics from metrics.csv
    metrics = pd.read_csv("metrics.csv")

    if len(metrics) < 2:
        return True

    # if latest is better than previous
    # replace model
    if metrics["loss"].iloc[-1] < metrics["loss"].iloc[-2]:
        return True

    return False


if __name__ == "__main__":
    # params for temp file name
    file_name = sys.argv[1]

    result = check_metrics()

    # if better than previous
    # make temp file
    if result:
        with open(file_name, "w") as file:
            file.write("temp")

    print(result)
