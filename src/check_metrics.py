import mlflow
import sys

def check_metrics():
    # get last two successful runs
    previous_run = mlflow.search_runs(
        experiment_ids="179663380575691319",
        filter_string="status = 'FINISHED'",
        max_results=2
    )
    
    # if there is no previous run
    # replace model
    if previous_run.shape[0] < 2:
        return True
    
    # if latest is better than previous
    # replace model
    if previous_run.iloc[0]['metrics.mse'] < previous_run.iloc[1]['metrics.mse']:
        return True
    
    return False

if __name__ == "__main__":
    
    # params for temp file name
    file_name = sys.argv[1]
    
    result = check_metrics()
    
    # save result to a temp file
    with open(file_name, 'w') as f:
        f.write(str(result))
    
    print(result)